import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from demo import synthetic_hexagons,split,models
class GroupedEvaluationTests(unittest.TestCase):
    def test_city_leakage_prevented(self):
        f=synthetic_hexagons();tr,te=split(f);self.assertFalse(set(f.city.iloc[tr])&set(f.city.iloc[te]));self.assertEqual(len(tr)+len(te),len(f))
    def test_preprocessing_learns_only_training_rows(self):
        f=synthetic_hexagons();tr,te=split(f);X=f.filter(regex='^feature_');m=models()['Logistic regression'];m.fit(X.iloc[tr],f.target.iloc[tr]);sc=m.named_steps['standardscaler']
        np.testing.assert_allclose(sc.mean_,X.iloc[tr].mean().values)
    def test_scores_finite_and_bounded(self):
        f=synthetic_hexagons();tr,te=split(f);X=f.filter(regex='^feature_');m=models()['Logistic regression'];m.fit(X.iloc[tr],f.target.iloc[tr]);p=m.predict_proba(X.iloc[te])[:,1];self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>=0)&(p<=1)).all())
if __name__=='__main__':unittest.main()
