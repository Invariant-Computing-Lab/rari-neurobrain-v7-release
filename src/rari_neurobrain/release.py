"""CP05 input validation and standalone saved-parameter inference. No network."""
import json
from pathlib import Path
import numpy as np
from .science import predict_models, infer_filter

def load_bank(path):
    p=Path(path)
    if not p.is_file() or p.stat().st_size>32*1024*1024: raise ValueError("MODEL_FILE")
    b=json.loads(p.read_text(encoding="utf8"))
    if not isinstance(b,dict) or not {"mean","basis","projection","models"}<=b.keys(): raise ValueError("MODEL_SCHEMA")
    n=len(b["mean"]);basis=np.asarray(b["basis"],float);proj=np.asarray(b["projection"],float)
    if n!=1212 or basis.shape!=(n,30) or proj.shape!=(n,30) or len(b["models"])!=522:raise ValueError("MODEL_DIMENSIONS")
    arrays=[np.asarray(b["mean"],float),basis,proj]
    tasks={"reconstruction":2,"hidden_filter_identification":4,"perturbation_prediction":16,"action_prediction":1,"cross_world_generalization":2,"recovery_attractor":2}
    for key,raw in b["models"].items():
        name,k,task=key.split("/");k=int(k)
        if name not in {"pca","classical_phase","quantum_phase"} or not 2<=k<=30 or task not in tasks:raise ValueError("MODEL_KEY")
        if np.asarray(raw["coef"]).shape!=(k+1,tasks[task]) or len(raw["mean"])!=k or len(raw["scale"])!=k:raise ValueError("COEFFICIENT_SHAPE")
        if np.any(np.asarray(raw["scale"])<=0):raise ValueError("MODEL_SCALE")
        arrays.extend([np.asarray(raw[a],float)for a in ["coef","mean","scale"]])
    if any(not np.isfinite(a).all()for a in arrays):raise ValueError("MODEL_NONFINITE")
    return b

def predict(bank,features,key="pca/12/reconstruction"):
    x=np.asarray(features,dtype=float)
    if x.ndim!=2 or x.shape[1]!=1212 or not 1<=len(x)<=10000:raise ValueError("FEATURE_SHAPE")
    if not np.isfinite(x).all():raise ValueError("FEATURE_NONFINITE")
    if key not in bank["models"]:raise ValueError("MODEL_KEY")
    return np.asarray(predict_models(bank,x)[key])

def fit_observer(u,y,alpha=1e-8):
    a,b=np.asarray(u,float),np.asarray(y,float)
    if a.ndim!=2 or a.shape[1]!=2 or b.shape!=a.shape or not 6<=len(a)<=10000:raise ValueError("PROBE_SHAPE")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(alpha) or alpha<=0:raise ValueError("PROBE_VALUES")
    return infer_filter(a,b,alpha=alpha)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("weights");p.add_argument("features");p.add_argument("--key",default="pca/12/reconstruction");a=p.parse_args()
    b=load_bank(a.weights);x=json.loads(Path(a.features).read_text());print(json.dumps({"key":a.key,"prediction":predict(b,x,a.key).tolist(),"scope":"synthetic finite estimator only"},allow_nan=False))
if __name__=="__main__":main()
