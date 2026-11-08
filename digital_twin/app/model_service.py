from __future__ import annotations
import hashlib
import json
import math
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / 'artifacts'

class TwinService:
    def __init__(self):
        self.manifest = json.loads((ART / 'workspace.json').read_text(encoding='utf-8'))
        self.meta = self.manifest['features']
        self.cols = list(self.meta)
        for name in ('model.joblib', 'hex_features.csv', 'original_values.csv', 'hex_centers.csv'):
            if hashlib.sha256((ART/name).read_bytes()).hexdigest() != self.manifest['sources'][name]['sha256']:
                raise ValueError('Model/data changed. Run prepare_workspace.py before starting.')
        for name in ('model_registry.json', 'model_scores.csv'):
            if name in self.manifest['sources'] and hashlib.sha256((ART/name).read_bytes()).hexdigest() != self.manifest['sources'][name]['sha256']:
                raise ValueError('Model registry changed. Run prepare_workspace.py before starting.')
        registry_path = ART / 'model_registry.json'
        self.registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else None
        self.models = {'lightgbm': joblib.load(ART / 'model.joblib')}
        if self.registry:
            for key, info in self.registry['models'].items():
                self.models[key] = joblib.load(ART / info['artifact'])
        for model in self.models.values():
            if hasattr(model, 'set_params'):
                try: model.set_params(n_jobs=1)
                except ValueError: pass
        self.model = self.models['lightgbm']
        self.df = pd.read_csv(ART / 'hex_features.csv').set_index('GRID_ID')
        self.raw = pd.read_csv(ART / 'original_values.csv').set_index('GRID_ID')
        self.centers = pd.read_csv(ART / 'hex_centers.csv').set_index('GRID_ID')
        self.model_scores = pd.read_csv(ART / 'model_scores.csv').set_index('GRID_ID') if (ART / 'model_scores.csv').exists() else None
        if not self.df.index.is_unique or not self.raw.index.is_unique or not self.centers.index.is_unique:
            raise ValueError('Artifact GRID_ID values must be unique')
        if set(self.raw.index) != set(self.df.index) or set(self.centers.index) != set(self.df.index):
            raise ValueError('Artifact GRID_ID mismatch')
        # Preserve the exact serialized training representation at tree thresholds.
        # A fresh division can differ by one floating-point unit from the CSV.
        self.scale_lookup = {}
        for col,meta in self.meta.items():
            if meta['editable']:
                self.scale_lookup[col] = dict(zip(self.raw[col], self.df.loc[self.raw.index,col]))
        self.geometry = {}
        authoritative = ART / 'authoritative_hexes.geojson'
        if authoritative.exists():
            data = json.loads(authoritative.read_text(encoding='utf-8'))
            for f in data['features']:
                gid = str(f['properties']['GRID_ID'])
                if gid in self.geometry or gid not in self.df.index or f['geometry']['type'] not in ('Polygon','MultiPolygon'):
                    raise ValueError('Invalid or duplicate authoritative geometry')
                def check_coords(v):
                    if isinstance(v[0], (int,float)):
                        if not all(math.isfinite(n) for n in v) or not (-180 <= v[0] <= 180 and -90 <= v[1] <= 90):
                            raise ValueError('Authoritative geometry must use WGS84 longitude/latitude')
                    else:
                        for x in v: check_coords(x)
                check_coords(f['geometry']['coordinates'])
                self.geometry[gid] = f['geometry']
        self.db_path = ART / 'scenarios.sqlite3'
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS scenarios (id TEXT PRIMARY KEY, created TEXT NOT NULL, body TEXT NOT NULL)')

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        try:
            with connection: yield connection
        finally:
            connection.close()

    def cities(self):
        return [dict(name=city, count=len(g), incidents=int(g.Total_Gun_Violence.sum()), score=float(g.twin_probability.mean()),
                     lat=float(self.centers.loc[g.index].lat.mean()), lon=float(self.centers.loc[g.index].lon.mean()))
                for city,g in self.df.groupby('City_name', sort=True)]

    def overview(self):
        models = self.registry['models'] if self.registry else {'lightgbm': {'name':'LightGBM','metrics':self.manifest['evaluation']}}
        return dict(self.manifest, models=models, cities=self.cities(), geometry=dict(verified_count=len(self.geometry),estimated_count=len(self.df)-len(self.geometry),
                    note='Original GIS polygons are not supplied. Unverified city records are shown without geographic placement; city summary positions remain approximate.'))

    def row(self, gid):
        if gid not in self.df.index: raise KeyError(gid)
        return self.df.loc[gid]

    def flags(self, gid):
        r = self.raw.loc[gid]
        flags = []
        if r.Total_Population == 0: flags.append('Population is zero in the source; population-based percentages were encoded as 0 where their denominator is zero.')
        if r.Total_Educational_Attainment_Base == 0: flags.append('Education denominator is zero; derived education percentages were encoded as 0.')
        if r.Poverty_Rate_Pct > 100: flags.append('Derived poverty percentage exceeds 100% in the original pipeline. Source numerator/denominator need review; the saved model input is preserved.')
        return flags

    def brief(self, gid):
        r = self.row(gid)
        return dict(id=gid, city=str(r.City_name), score=float(r.twin_probability), incidents=int(r.Total_Gun_Violence),
                    population=float(self.raw.loc[gid,'Total_Population']), split=str(self.raw.loc[gid,'evaluation_split']),quality_flags=self.flags(gid))

    def list_hexes(self, city=None, q='', sort='score', offset=0, limit=60):
        frame = self.df
        if city:
            if city not in set(self.df.City_name): raise ValueError('Unknown city')
            frame = frame[frame.City_name == city]
        if q: frame = frame[frame.index.str.contains(q.strip(),case=False,regex=False)]
        if sort == 'id': frame = frame.sort_index()
        else: frame = frame.sort_values('Total_Gun_Violence' if sort == 'incidents' else 'twin_probability', ascending=False, kind='stable')
        return dict(total=len(frame), items=[self.brief(gid) for gid in frame.index[offset:offset+limit]])

    def map_hexes(self, city):
        if city not in set(self.df.City_name): raise ValueError('Unknown city')
        features = []
        for gid,r in self.df[self.df.City_name == city].iterrows():
            geometry = self.geometry.get(gid)
            if geometry is None:
                c = self.centers.loc[gid]
                geometry = dict(type='Point',coordinates=[float(c.lon),float(c.lat)])
            features.append(dict(type='Feature',geometry=geometry,properties=dict(id=gid,score=float(r.twin_probability),incidents=int(r.Total_Gun_Violence),verified=gid in self.geometry)))
        return dict(type='FeatureCollection',features=features)

    def explain(self, values):
        c = self.model.booster_.predict(pd.DataFrame([values],columns=self.cols),pred_contrib=True,num_threads=1)[0]
        order = np.argsort(np.abs(c[:-1]))[::-1][:8]
        return [dict(feature=self.cols[i],label=self.meta[self.cols[i]]['label'],contribution=float(c[i])) for i in order]

    def detail(self, gid):
        r = self.row(gid)
        result = self.brief(gid)
        model_scores = {}
        if self.model_scores is not None:
            for key in self.models:
                if key in self.model_scores: model_scores[key] = float(self.model_scores.loc[gid,key])
        result.update(values={c:float(self.raw.loc[gid,c]) for c in self.cols}, model_scores=model_scores,
                      contributions=self.explain({c:float(r[c]) for c in self.cols}),
                      geometry_verified=gid in self.geometry,
                      city_percentile=float((self.df.loc[self.df.City_name==r.City_name,'twin_probability'] <= r.twin_probability).mean()*100))
        return result

    def score(self, gid, overrides, model_key='lightgbm'):
        if model_key not in self.models: raise ValueError(f'Unknown model: {model_key}')
        row = self.row(gid)
        values = {c:float(self.raw.loc[gid,c]) for c in self.cols}
        scaled = {c:float(row[c]) for c in self.cols}
        changes, flags = [], self.flags(gid)
        for key,value in overrides.items():
            if key not in self.meta: raise ValueError(f'Unknown feature: {key}')
            m = self.meta[key]
            if not m['editable']: raise ValueError(f'{m["label"]} is context only')
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value): raise ValueError(f'Invalid value for {key}')
            value = float(value)
            scenario_max = max(float(m.get('scenario_max', m['max'])), float(self.raw.loc[gid, key]))
            scenario_min = min(float(m.get('scenario_min', m['min'])), float(self.raw.loc[gid, key]))
            if not scenario_min <= value <= scenario_max: raise ValueError(f'{m["label"]} must be between {scenario_min:g} and {scenario_max:g} for this hexagon')
            if m.get('facility', False) and not value.is_integer(): raise ValueError(f'{m["label"]} requires a whole-number count')
            values[key] = value
            before = float(self.raw.loc[gid,key])
            if value != before:
                scaled[key] = self.scale_lookup[key].get(value, (value-m['min'])/(m['max']-m['min']) if m['max']>m['min'] else 0)
                changes.append(dict(feature=key,label=m['label'],before=before,after=value,unit=m['unit']))
                if value > m['q95']: flags.append(f'{m["label"]} exceeds the study-wide 95th percentile. This is an uncommon scenario input.')
        if changes:
            editable = [c for c in self.cols if self.meta[c]['editable']]
            city = self.df[self.df.City_name == row.City_name]
            distance = np.abs(city[editable].to_numpy()-np.array([scaled[c] for c in editable])).mean(axis=1)
            nearest = float(distance.min())
        else: nearest = 0.
        model = self.models[model_key]
        predict_kwargs = {'num_threads': 1} if model_key == 'lightgbm' else {}
        score = float(model.predict_proba(pd.DataFrame([scaled], columns=self.cols), **predict_kwargs)[0,1])
        baseline = float(self.model_scores.loc[gid, model_key]) if self.model_scores is not None and model_key in self.model_scores else float(row.twin_probability)
        return dict(id=gid,city=str(row.City_name),baseline=baseline,score=score,delta=score-baseline,changes=changes,
                    model=model_key,baseline_model_score=baseline,values=values,warnings=flags,nearest_city_distance=nearest,model_version=self.manifest['model_version'],data_version=self.manifest['data_version'])

    def sensitivity(self, gid, feature, overrides, model_key='lightgbm'):
        if feature not in self.meta or not self.meta[feature].get('editable', False): raise ValueError('Choose a scenario feature')
        self.score(gid, overrides, model_key)
        m=self.meta[feature]
        current=float(overrides.get(feature,float(self.raw.loc[gid,feature])))
        base=float(self.raw.loc[gid,feature])
        lo=float(m.get('boundary_min',m.get('scenario_min',m['min']))); hi=float(m.get('boundary_max',m.get('scenario_max',m['max'])))
        current=min(hi,max(lo,current))
        if m.get('facility',False):
            low=max(lo,current-5); high=min(hi,current+5)
            vals=list(range(int(math.ceil(low)),int(math.floor(high))+1))
        else:
            low=max(lo,current-(hi-lo)*.1); high=min(hi,current+(hi-lo)*.1)
            vals=list(np.linspace(low,high,21))
        points=[dict(value=float(v),score=self.score(gid,dict(overrides,**{feature:float(v)}),model_key)['score']) for v in vals]
        return dict(feature=feature,model=model_key,points=points,note='One feature varies; all other current scenario inputs stay fixed. These are model responses, not effects over time.')

    def optimize(self, gid, model_key='lightgbm', max_changes=22):
        if model_key not in self.models: raise ValueError(f'Unknown model: {model_key}')
        max_changes=max(1,min(int(max_changes),len(self.cols)))
        baseline=self.score(gid,{},model_key)
        chosen={}; used=set(); changes=[]; trajectory=[dict(changes=0,score=baseline['score'])]
        for _ in range(max_changes):
            best=None
            for feature in self.cols:
                if feature in used: continue
                m=self.meta[feature]; base=float(self.raw.loc[gid,feature])
                lo=float(m.get('boundary_min',m.get('scenario_min',m['min']))); hi=float(m.get('boundary_max',m.get('scenario_max',m['max'])))
                vals=[lo,hi] if hi>lo else [base]
                if m.get('facility',False): vals=[int(round(v)) for v in vals]
                for value in dict.fromkeys(vals):
                    trial=dict(chosen); trial[feature]=value
                    result=self.score(gid,trial,model_key)
                    if best is None or result['score'] < best['result']['score']:
                        best=dict(feature=feature,value=float(value),result=result)
            if best is None or best['result']['score'] >= trajectory[-1]['score']-1e-10: break
            feature=best['feature']; value=best['value']; chosen[feature]=int(value) if self.meta[feature].get('facility',False) else value; used.add(feature)
            before=float(self.raw.loc[gid,feature]); m=self.meta[feature]
            changes.append(dict(feature=feature,label=m['label'],before=before,after=chosen[feature],unit=m['unit'],score=best['result']['score']))
            trajectory.append(dict(changes=len(changes),score=best['result']['score']))
        final=self.score(gid,chosen,model_key)
        return dict(id=gid,model=model_key,baseline=baseline['score'],optimized_score=final['score'],delta=final['score']-baseline['score'],overrides=chosen,changes=changes,trajectory=trajectory,result=final,interpretation='Greedy search of boundary values from the baseline; not a guaranteed global minimum. Bounds use the positive-support 5th percentile when a variable has zero inflation, and the study-wide 95th percentile as the upper limit. This is a sensitivity result, not a causal intervention recommendation.')

    def save(self, name, gid, overrides, model_key='lightgbm'):
        result=self.score(gid,overrides,model_key)
        record=dict(result,id=str(uuid.uuid4()),grid_id=gid,name=name.strip(),overrides=overrides,created=datetime.now(timezone.utc).isoformat())
        if not record['name']: raise ValueError('Give the scenario a name')
        with self.db() as db: db.execute('INSERT INTO scenarios VALUES (?,?,?)',(record['id'],record['created'],json.dumps(record)))
        return record

    def scenarios(self):
        with self.db() as db: rows=db.execute('SELECT body FROM scenarios ORDER BY created DESC').fetchall()
        return [json.loads(r[0]) for r in rows]

    def delete(self, sid):
        with self.db() as db:
            if db.execute('DELETE FROM scenarios WHERE id=?',(sid,)).rowcount == 0: raise KeyError(sid)

_service = None
_lock = threading.Lock()
def get_service():
    global _service
    with _lock:
        if _service is None: _service = TwinService()
    return _service
