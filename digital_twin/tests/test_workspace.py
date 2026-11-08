"""Integration and data invariants. All temporary writes stay in digital_twin."""
import hashlib
import json
import math
import sys
import uuid
import unittest
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from digital_twin.app.model_service import TwinService

class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.s=TwinService()

    def test_all_baselines_from_original_units(self):
        s=self.s
        transformed=pd.DataFrame({c:(s.raw[c]-m['min'])/(m['max']-m['min']) if m['max']>m['min'] else s.raw[c]*0 for c,m in s.meta.items()})
        np.testing.assert_allclose(transformed.loc[s.df.index,s.cols],s.df[s.cols],atol=1e-12,rtol=0)
        # Original-unit equivalence holds to floating precision. Baseline scoring
        # deliberately preserves serialized inputs at exact tree thresholds.
        canonical=s.df[s.cols].copy()
        for col,lookup in s.scale_lookup.items(): canonical[col]=s.raw.loc[canonical.index,col].map(lookup)
        predictions=s.model.predict_proba(canonical)[:,1]
        np.testing.assert_allclose(predictions,s.df.loc[transformed.index,'twin_probability'],atol=1e-10,rtol=0)

    def test_population_and_poverty_flags(self):
        s=self.s
        for gid in s.raw.index[s.raw.Poverty_Rate_Pct>100]: self.assertTrue(any('exceeds 100%' in f for f in s.flags(gid)))
        self.assertEqual(sum(s.raw.Poverty_Rate_Pct>100),17)
        self.assertEqual(sum(s.raw.Total_Population==0),2792)

    def test_full_large_city_pagination_and_literal_search(self):
        s=self.s;ids=[]
        for offset in range(0,1136,60): ids.extend(x['id'] for x in s.list_hexes('Houston',offset=offset)['items'])
        self.assertEqual(len(ids),1136);self.assertEqual(len(set(ids)),1136)
        self.assertEqual(s.list_hexes(q='[')['total'],0)
        self.assertEqual(len(s.map_hexes('Houston')['features']),1136)
        self.assertTrue(all(f['geometry']['type']=='Point' for f in s.map_hexes('Chicago')['features']))

    def test_scores_and_invalid_overrides(self):
        s=self.s;gid=s.df.index[0]
        self.assertAlmostEqual(s.score(gid,{})['score'],s.df.loc[gid,'twin_probability'],places=12)
        for bad in [{'Schools':-1},{'Schools':999},{'Schools':.5},{'Schools':float('nan')},{'Schools':float('inf')},{'Schools':True},{'no_such_feature':1},{'Black_Percentage':-1}]:
            with self.subTest(bad=bad),self.assertRaises(ValueError): s.score(gid,bad)
        contextual=s.score(gid,{'Black_Percentage':float(s.raw.loc[gid,'Black_Percentage'])+.01})
        self.assertEqual(contextual['changes'][0]['feature'],'Black_Percentage')
        with self.assertRaises(KeyError): s.score('not-real',{})

    def test_five_model_registry_and_scenario_scores(self):
        s=self.s
        self.assertEqual(set(s.models), {'lightgbm','logistic_regression','random_forest','gradient_boosting','xgboost'})
        gid=s.df.index[0]; detail=s.detail(gid)
        self.assertEqual(set(detail['model_scores']), set(s.models))
        for key in s.models:
            result=s.score(gid,{'Schools':1},key)
            self.assertEqual(result['model'],key)
            self.assertTrue(0 <= result['score'] <= 1)

    def test_optimizer_lowers_or_matches_baseline(self):
        s=self.s; gid=s.df.index[0]
        result=s.optimize(gid,'lightgbm')
        self.assertLessEqual(result['optimized_score'],result['baseline']+1e-12)
        self.assertLessEqual(len(result['changes']),22)
        for key,value in result['overrides'].items():
            self.assertGreaterEqual(float(value), float(s.meta[key]['boundary_min']))
            self.assertLessEqual(float(value), float(s.meta[key]['boundary_max']))

    def test_sensitivity_holds_other_inputs(self):
        s=self.s;gid=s.df.index[100];o={'Schools':3,'Community_Center':1}
        result=s.sensitivity(gid,'Schools',o)
        for p in result['points']:
            expected=s.score(gid,dict(o,Schools=p['value']))
            self.assertAlmostEqual(p['score'],expected['score'],places=12)

    def test_optimizer_snapshot_matches_rescoring(self):
        s=self.s;gid=s.df.index[0]
        result=s.optimize(gid,'random_forest',max_changes=2)
        expected=s.score(gid,result['overrides'],'random_forest')
        self.assertEqual(result['result'],expected)
        self.assertEqual(result['optimized_score'],expected['score'])

    def test_sensitivity_outside_empirical_bounds(self):
        s=self.s
        for feature in ['Schools','Black_Segregation_Index']:
            if feature not in s.meta: continue
            hi=s.meta[feature]['boundary_max']
            candidates=s.raw.index[s.raw[feature]>hi]
            if not len(candidates): continue
            result=s.sensitivity(candidates[0],feature,{},'random_forest')
            self.assertTrue(result['points'])
            for point in result['points']:
                self.assertLessEqual(point['value'],hi)
                self.assertGreaterEqual(point['value'],s.meta[feature]['boundary_min'])

    def test_saved_snapshot_roundtrip(self):
        s=self.s;original=s.db_path
        test_path=ROOT/'digital_twin/tests'/f'test-{uuid.uuid4()}.sqlite3'
        try:
                s.db_path=test_path
                with s.db() as db: db.execute('CREATE TABLE scenarios(id TEXT PRIMARY KEY,created TEXT,body TEXT)')
                record=s.save('Test snapshot',s.df.index[0],{'Schools':2})
                loaded=s.scenarios()[0]
                self.assertEqual(record,loaded)
                self.assertEqual(loaded['model_version'],s.manifest['model_version'])
                self.assertEqual(loaded['values']['Schools'],2)
                s.delete(record['id']);self.assertEqual(s.scenarios(),[])
        finally:
            s.db_path=original
            if test_path.exists(): test_path.unlink()

    def test_explanations_add_to_model_log_odds(self):
        s=self.s;x=s.df[s.cols].iloc[:35]
        contributions=s.model.booster_.predict(x,pred_contrib=True,num_threads=1)
        probabilities=1/(1+np.exp(-contributions.sum(axis=1)))
        np.testing.assert_allclose(probabilities,s.model.predict_proba(x)[:,1],atol=1e-10)
        self.assertEqual(len(s.detail(s.df.index[10])['contributions']),8)

    def test_research_files_unchanged(self):
        manifest=json.loads((ROOT/'digital_twin/artifacts/protected_files.json').read_text())
        for name,digest in manifest.items():
            with self.subTest(file=name):self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)

class HTTPTests(unittest.TestCase):
    base='http://127.0.0.1:8765'
    def request(self,path,body=None,headers=None):
        req=Request(self.base+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json',**(headers or {})})
        try:
            with urlopen(req,timeout=20) as r:return r.status,json.load(r)
        except HTTPError as e:return e.code,json.load(e)

    def test_health_and_overview(self):
        status,h=self.request('/api/health');self.assertEqual(status,200);self.assertEqual(h['n_hexes'],11448)
        status,o=self.request('/api/overview');self.assertEqual(len(o['cities']),35)
        self.assertEqual(o['total_incidents'],2618)

    def test_reject_malformed_and_cross_origin(self):
        _,h=self.request('/api/hexes?city=Chicago&limit=1');gid=h['items'][0]['id']
        for body in [{'grid_id':gid,'overrides':{'Schools':True}},{'grid_id':gid,'overrides':{'Schools':'2'}},{'grid_id':gid,'unknown':1}]:
            self.assertEqual(self.request('/api/predict',body)[0],422)
        self.assertEqual(self.request('/api/predict',{'grid_id':gid,'overrides':{'Schools':-1}})[0],400)
        self.assertEqual(self.request('/api/predict',{'grid_id':'NOT-REAL'})[0],404)
        self.assertEqual(self.request('/api/scenarios',{'grid_id':gid,'name':'test'},headers={'Origin':'https://unrelated.example'})[0],403)
        self.assertEqual(self.request('/api/hexes?limit=0')[0],422)

    def test_api_score_uses_original_count(self):
        _,h=self.request('/api/hexes?city=Chicago&limit=1');gid=h['items'][0]['id']
        _,d=self.request('/api/hex/'+gid)
        status,s=self.request('/api/predict',{'grid_id':gid,'overrides':{'Schools':3}})
        self.assertEqual(status,200);self.assertEqual(s['values']['Schools'],3)
        self.assertAlmostEqual(s['baseline'],d['score'],places=12)
        self.assertEqual(self.request('/api/hexes?q=%5B')[0],200)

if __name__=='__main__':unittest.main(verbosity=2)

