#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,pathlib,sys
import numpy as np
TASKS=('reconstruction','hidden_filter_identification','perturbation_prediction','action_prediction','cross_world_generalization','recovery_attractor')
EPS=1e-8

def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256();
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def req(c,m):
 if not c: raise RuntimeError(m)
def rows(root,split):return [read(p) for p in sorted((root/'private/banks'/split).glob('*/COMPLETE.json'))]
def chosen(rs):return sorted([r for r in rs if r['policy']=='balanced' and r['observer']==0],key=lambda r:r['family'])
def x_of(rs):return np.asarray([r['features'] for r in chosen(rs)],float)
def y_of(rs):return {t:np.asarray([r['targets'][t] for r in chosen(rs)],float) for t in TASKS}
def mse_rows(p,y):return np.mean((np.asarray(p,float)-np.asarray(y,float))**2,axis=1)
def design_mats(xd,xv,xn):
 mu=xd.mean(0);_,_,vt=np.linalg.svd(xd-mu,full_matrices=False);basis=vt[:30].T
 rng=np.random.default_rng(7102);projection=rng.normal(size=(xd.shape[1],30))/np.sqrt(xd.shape[1])
 def maps(x):
  z=x@projection;return {'pca':(x-mu)@basis,'classical_phase':np.cos(z+.7)**2,'quantum_phase':np.cos(z)**2}
 md,mv,mn=maps(xd),maps(xv),maps(xn);pre={}
 for name in ('pca','classical_phase','quantum_phase'):
  for k in range(2,31):
   d=md[name][:,:k];mean=d.mean(0);scale=np.maximum(d.std(0),1e-8)
   zd=np.column_stack([np.ones(len(d)),(d-mean)/scale]);zv=np.column_stack([np.ones(len(xv)),(mv[name][:,:k]-mean)/scale]);zn=np.column_stack([np.ones(len(xn)),(mn[name][:,:k]-mean)/scale])
   pen=np.eye(zd.shape[1])*.1;pen[0,0]=0;H=np.linalg.solve(zd.T@zd+pen,zd.T);pre[f'{name}/{k}']=(H,zv,zn)
 return pre

def main():
 root=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]) if len(sys.argv)>2 else root/'NULL_REPLAY_SELFTEST.json'
 splits=read(root/'private/vault/SPLITS.json');req(set(splits)=={'development','validation','calibration','closed_test','independent_repeat','null'},'SPLIT_KEYS')
 data={s:rows(root,s) for s in ('development','validation','null')};req({k:len(v) for k,v in data.items()}=={'development':256,'validation':128,'null':256},'INPUT_COUNTS')
 for split,rs in data.items():req(sorted({r['family'] for r in rs})==sorted(splits[split]),'SPLIT_RECORD_MEMBERSHIP:'+split)
 xd,xv,xn=(x_of(data[s]) for s in ('development','validation','null'));yd,yv,yn=(y_of(data[s]) for s in ('development','validation','null'));scales=read(root/'private/TARGET_SCALES.json')
 pre=design_mats(xd,xv,xn);reported=read(root/'results/NULL_AND_MULTIPLICITY.json')['pipeline'];receipts=[read(root/p) for p in reported['refit_receipts']];req(len(receipts)==199,'RECEIPTS')
 rng=np.random.default_rng(9327);maxdiff=0.;sels=[];scores=[]
 for rep in range(199):
  pd,pv,pn=(rng.permutation(n) for n in (len(xd),len(xv),len(xn)));group_scores={};coefs={}
  for key,(H,zv,zn) in pre.items():
   taskvals=[];taskcoef={}
   for t in TASKS:
    coef=H@yd[t][pd];taskcoef[t]=coef;taskvals.append(mse_rows(zv@coef,yv[t][pv])/max(float(scales[t]),EPS))
   group_scores[key]=float(np.mean(np.stack(taskvals),axis=0).mean());coefs[key]=taskcoef
  selected=min(group_scores,key=lambda key:(group_scores[key],int(key.split('/')[1]),key));H,zv,zn=pre[selected];null_task=[]
  for t in TASKS:null_task.append(mse_rows(zn@coefs[selected][t],yn[t][pn])/max(float(scales[t]),EPS))
  score=float(np.mean(np.stack(null_task),axis=0).mean());rr=receipts[rep];diff=abs(score-float(rr['score']));maxdiff=max(maxdiff,diff);req(selected==rr['selected'],'SELECTION_MISMATCH:'+str(rep));req(diff<=1e-9*max(1.,abs(score),abs(float(rr['score']))),'SCORE_MISMATCH:'+str(rep));sels.append(selected);scores.append(score)
 result={'status':'PASS','repeats':199,'seed':9327,'input_counts':{k:len(v) for k,v in data.items()},'selection_matches':199,'max_score_abs_error':maxdiff,'selected_sha256':hashlib.sha256((json.dumps(sels,separators=(',',':'))+'\n').encode()).hexdigest(),'scores_sha256':hashlib.sha256((json.dumps(scores,separators=(',',':'))+'\n').encode()).hexdigest(),'implementation':'independent matrix replay; no production scoring/null imports'}
 out.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n');print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
