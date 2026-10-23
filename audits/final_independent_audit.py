"""Independent scoring arithmetic, Python standard library only; no model/provider."""
import json, math, pathlib, sys

def read(p):return json.loads(p.read_text())
def check(root):
    max_error=0.; checked=0
    scoring=root/'private/scoring' if (root/'private/scoring').is_dir() else root/'evaluation'
    results=root/'results' if (root/'results').is_dir() else root/'metrics'
    for split in ('closed_test','independent_repeat'):
        data=read(scoring/split/'audit_data.json')
        result=read(results/(split.upper()+'_RESULTS.json'))
        for key,predictions in data['predictions'].items():
            targets=data['targets'][key.split('/')[-1]]
            if len(predictions)!=len(targets):raise ValueError('AUDIT_ROW_COUNT')
            squared=[]
            for a,b in zip(predictions,targets):
                if len(a)!=len(b):raise ValueError('AUDIT_TARGET_DIMENSION')
                squared.extend((float(x)-float(y))**2 for x,y in zip(a,b))
            mse=math.fsum(squared)/len(squared); reported=result['scores'][key]
            error=abs(mse-reported);max_error=max(max_error,error)
            if error>1e-10*max(1,abs(mse),abs(reported)):raise ValueError('MSE_AUDIT_MISMATCH:'+key)
            checked+=1
    primary=read(results/'PRIMARY_RESULTS.json')
    draw=read(scoring/'PRIMARY_ARITHMETIC.json')
    raw=[];ambiguous=0
    for index,values in enumerate(draw['contrast_vectors']):
        observed=math.fsum(values)/len(values);count=0
        for signs in draw['signs']:
            z=math.fsum(float(x)*s for x,s in zip(values,signs))/len(values)
            count+=z>=observed
        recorded=draw['counts'][index]
        # Tiny summation-order ties are disclosed, not silently promoted.
        if count!=recorded:ambiguous+=1
        raw.append((recorded+1)/(len(draw['signs'])+1))
    n=len(raw);ordered=sorted(range(n),key=lambda k:raw[k]);adj=[0.]*n;running=0.
    for i,k in enumerate(ordered):running=max(running,min(1.,(n-i)*raw[k]));adj[k]=running
    if any(abs(x-y)>1e-12 for x,y in zip(raw,primary['raw_p'])):raise ValueError('PLUS_ONE_AUDIT_MISMATCH')
    if any(abs(x-y)>1e-12 for x,y in zip(adj,primary['holm_p'])):raise ValueError('HOLM_AUDIT_MISMATCH')
    return {'status':'PASS' if ambiguous==0 else 'REQUIRES_FLOAT_TIE_REVIEW','mse_count':checked,'max_mse_abs_error':max_error,
            'primary_tests':n,'sign_flip_summation_tie_discrepancies':ambiguous,
            'implementation':'independent Python stdlib fsum and ordered Holm; no production scoring import'}
if __name__=='__main__':
    print(json.dumps(check(pathlib.Path(sys.argv[1])),sort_keys=True))
