"""Grouped ML demonstration on synthetic hexagons, separate from historical study."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def synthetic_hexagons(seed=42):
    rng=np.random.default_rng(seed);n=1050;X=rng.normal(size=(n,22));city=np.repeat(np.arange(35),30)
    logit=-1+X[:,0]+.6*X[:,1]-.4*X[:,2]+rng.normal(0,.5,35)[city]
    y=(rng.random(n)<1/(1+np.exp(-logit))).astype(int)
    return pd.DataFrame({'hex_id':[f'SYN-{i:04}' for i in range(n)],'city':city,**{f'feature_{j+1}':X[:,j] for j in range(22)},'target':y})
def split(frame):
    return next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(frame,frame.target,groups=frame.city))
def models():
    return {'Logistic regression':make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=1000,random_state=42)),
        'Random forest':make_pipeline(SimpleImputer(),RandomForestClassifier(n_estimators=100,max_depth=7,random_state=42,n_jobs=1)),
        'Gradient boosting':make_pipeline(SimpleImputer(),GradientBoostingClassifier(n_estimators=75,max_depth=2,random_state=42))}
def main():
    f=synthetic_hexagons();f.to_csv(ROOT/'data/sample/hexagons.csv',index=False);tr,te=split(f);X=f.filter(regex='^feature_');rows=[]
    for name,m in models().items():
        m.fit(X.iloc[tr],f.target.iloc[tr]);p=m.predict_proba(X.iloc[te])[:,1]
        rows.append({'model':name,'roc_auc':roc_auc_score(f.target.iloc[te],p),'pr_auc':average_precision_score(f.target.iloc[te],p),'f1_at_0_5':f1_score(f.target.iloc[te],p>=.5)})
    out=ROOT/'outputs/demo';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    meta={'data':'SYNTHETIC','rows':len(f),'features':22,'train_cities':sorted(map(int,set(f.city.iloc[tr]))),'test_cities':sorted(map(int,set(f.city.iloc[te]))),'threshold':.5,'models':rows}
    (out/'run.json').write_text(json.dumps(meta,indent=2))
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');pd.DataFrame(rows).set_index('model')[['roc_auc','pr_auc','f1_at_0_5']].plot.barh(ax=ax,color=['#137c8b','#356db3','#c75b40']);ax.set(xlim=(0,1),xlabel='Metric on synthetic held out cities',title='Grouped holdout demonstration | synthetic data');fig.savefig(ROOT/'docs/assets/demo.png',dpi=160);plt.close(fig)
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
