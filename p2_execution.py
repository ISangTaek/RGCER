"""P2 smoke release identity and one independent validation audit."""
from pathlib import Path
import hashlib
import math
import os
import re
from p2_contract import require, bound_json, PROTOCOL_SHA, MEMBERS_SHA, PANELS, ROUTES, METHODS
from p2_data import digest

APPROVAL_SHA='3c8143e31eb66cba3d4ed25003bdd2fa72b276429f552b312042bdddad409bdb'


def smoke_matrix():
    rows=[]
    for panel in PANELS:
        for route in ROUTES:
            rows.append(dict(run_id=f'P2SMOKE_{panel}_{route}_source',role='source',panel=panel,route=route,seed=42,updates=1))
    for method in METHODS:
        rows.append(dict(run_id=f'P2SMOKE_{method}',role='target',panel='mouse',route='oral',seed=42,
                         method=method,source_run_id='P2SMOKE_mouse_oral_source',updates=36))
    return rows


def load_release(path,sha,approval):
    a=bound_json(approval,APPROVAL_SHA)
    require(a['budget_status']=='APPROVED','budget not approved')
    r=bound_json(path,sha)
    require(set(r)=={'schema','stage','commit','protocol_sha256','members_sha256','approval_sha256',
                    'updates_max','runs','test_authorized','calibration_authorized','retry_authorized'},'release schema')
    require(r['schema']=='p2_release_v1' and r['stage']=='SMOKE','only smoke release supported')
    require(isinstance(r['commit'],str) and re.fullmatch('[0-9a-f]{40}',r['commit']),'release commit')
    require(r['protocol_sha256']==PROTOCOL_SHA and r['members_sha256']==MEMBERS_SHA and r['approval_sha256']==APPROVAL_SHA,'release inputs')
    require(type(r['updates_max']) is int and r['updates_max']==112,'smoke budget')
    require(digest(r['runs'])==digest(smoke_matrix()),'smoke matrix')
    require(all(r[k] is False for k in ('test_authorized','calibration_authorized','retry_authorized')),'forbidden permissions')
    return r


def write_new(path,value):
    import json
    raw=json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
    with Path(path).open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


def audit_validation(history,view,best_epoch):
    """Recompute from observations, independently of the engine metrics function."""
    require(history,'empty validation history')
    expected={(t,r['sample_id']):r for t,rows in view.rows['validation'].items() for r in rows}
    scores=[]
    for epoch,h in enumerate(history):
        require(type(h['epoch']) is int and h['epoch']==epoch,'audit epoch sequence')
        rows=h['validation_rows'];keys=[(r['task'],r['sample_id']) for r in rows]
        require(len(keys)==len(set(keys)) and set(keys)==set(expected),'audit validation members')
        values={t:[] for t in view.tasks}
        for r in rows:
            e=expected[(r['task'],r['sample_id'])]
            require(r['split']=='validation' and all(r[k]==e[k] for k in ('label','canonical','group')),'audit labels/groups/split')
            require(type(r['prediction']) in (int,float) and math.isfinite(r['prediction']),'audit finite prediction')
            values[r['task']].append((r['label'],r['prediction']))
        per={}
        for t,pairs in values.items():
            n=len(pairs);m=math.fsum(y for y,_ in pairs)/n
            sse=math.fsum((y-p)**2 for y,p in pairs);sst=math.fsum((y-m)**2 for y,_ in pairs)
            per[t]=dict(n=n,rmse=math.sqrt(sse/n),mae=math.fsum(abs(y-p) for y,p in pairs)/n,
                        r2=None if sst==0 else 1-sse/sst,r2_reason='constant_labels' if sst==0 else None)
        macro=math.fsum(v['rmse'] for v in per.values())/len(per)
        reported=h['validation'];require(set(reported['endpoints'])==set(per),'audit endpoint set')
        for t,v in per.items():
            actual=reported['endpoints'][t]
            require(set(actual)==set(v),'audit endpoint fields')
            for k,x in v.items():
                y=actual[k]
                if k in ('rmse','mae','r2') and x is not None:
                    require(type(y) in (int,float) and math.isclose(y,x,rel_tol=1e-10,abs_tol=1e-12),'audit metric mismatch')
                else:require(type(y) is type(x) and y==x,'audit metadata mismatch')
        require(type(reported['macro_rmse']) in (int,float) and math.isclose(reported['macro_rmse'],macro,rel_tol=1e-10,abs_tol=1e-12),'audit macro mismatch')
        scores.append(reported['macro_rmse'])
    require(type(best_epoch) is int and best_epoch==min(range(len(scores)),key=scores.__getitem__),'audit best selection')
    return dict(status='PASS',epochs=len(history),observations_per_epoch=len(expected),best_epoch=best_epoch)
