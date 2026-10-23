import unittest,json,copy,tempfile
from pathlib import Path
import numpy as np
from rari_neurobrain.release import load_bank,predict,fit_observer
from rari_neurobrain.evaluation import conformal_radius,holm
ROOT=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.b=load_bank(ROOT/"assets/MODELS.json");cls.x=json.loads((ROOT/"assets/example_features.json").read_text());cls.expected=json.loads((ROOT/"assets/example_expected.json").read_text())
 def test_all_522_saved_parameter_predictions(self):
  from rari_neurobrain.science import predict_models
  y=predict_models(self.b,self.x)
  for k,v in y.items(): np.testing.assert_allclose(v,self.expected[k],rtol=1e-12,atol=1e-12)
 def test_reload_identical(self):np.testing.assert_array_equal(predict(self.b,self.x),predict(load_bank(ROOT/"assets/MODELS.json"),self.x))
 def test_wrong_width(self):
  with self.assertRaises(ValueError):predict(self.b,[[0]*1211])
 def test_wrong_rank(self):
  with self.assertRaises(ValueError):predict(self.b,[0]*1212)
 def test_empty(self):
  with self.assertRaises(ValueError):predict(self.b,np.empty((0,1212)))
 def test_nonfinite(self):
  with self.assertRaises(ValueError):predict(self.b,np.full((1,1212),np.nan))
 def test_unknown_key(self):
  with self.assertRaises(ValueError):predict(self.b,self.x,"pca/31/reconstruction")
 def test_bad_model_schema(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/"x.json";p.write_text("{}");
   with self.assertRaises(ValueError):load_bank(p)
 def test_filter_reproduction(self):
  d=json.loads((ROOT/"assets/example_filter_input.json").read_text());e=json.loads((ROOT/"assets/example_filter_expected.json").read_text());r=fit_observer(d["u"],d["y"]);self.assertEqual(r["mode"],e["mode"]);np.testing.assert_allclose(r["A"],e["A"],rtol=1e-12,atol=1e-12)
 def test_filter_shape(self):
  with self.assertRaises(ValueError):fit_observer([[0,0]]*5,[[0,0]]*5)
 def test_filter_nan(self):
  with self.assertRaises(ValueError):fit_observer([[0,0]]*6,[[float("nan"),0]]*6)
 def test_filter_zero_alpha(self):
  with self.assertRaises(ValueError):fit_observer([[0,0]]*6,[[0,0]]*6,0)
 def test_constant_response_not_parameter_rank(self):
  from rari_neurobrain.science import interventions
  r=fit_observer(interventions("balanced",12,0),np.zeros((12,2)));self.assertEqual(r["rank"],0);self.assertEqual(r["parameter_identifiability"]["J_P_rank"],6)
 def test_empty_common_not_success(self):
  a=np.diag([1,0]);b=np.diag([0,1]);_,s,v=np.linalg.svd(np.vstack([np.eye(2)-a,np.eye(2)-b]));self.assertEqual(v[s<1e-6].shape[0],0)
 def test_conformal_order_30(self):self.assertEqual(conformal_radius(np.arange(32),.1),29)
 def test_unbounded_small_calibration(self):self.assertIsNone(conformal_radius([1],.1))
 def test_holm(self):np.testing.assert_allclose(holm([.0005]*4+[.9]*8),[.006]*4+[1]*8)
if __name__=="__main__":unittest.main()
