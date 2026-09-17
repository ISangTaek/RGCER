"""S5C selected-best inference; all validation regressions precede any test.

No optimizer, training, checkpoint selection, or conformal calibration here.
"""
import hashlib,io,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from dataset115_contract import PRIMARY,semantic_digest
from dataset115_training import require
from dataset115_smoke import state_digest
from dataset115_adapter import GraphTaskView,TrainOnlyScaler
from dataset115_model import build_human5_model
from architecture.prediction_heads import decode_prediction

LOCK_SHA='33a32f253341f0b4b590982beb738bb244ebf5422ec4fc281a8a66ab1a392204'
MATRIX={(m,s) for m in ('B0','B1','RPT') for s in range(42,47)}

def strict_json(raw):
    def bad(x):raise ValueError(x)
    def pairs(items):
        d={}
        for k,v in items:
            require(k not in d,'duplicate JSON key');d[k]=v
        return d
    return json.loads(raw,parse_constant=bad,object_pairs_hook=pairs)

def load_lock(path):
    raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==LOCK_SHA,'unapproved selection lock')
    d=strict_json(raw)
    require(d['schema']=='s5c_route_b_selected_v1' and len(d['runs'])==15 and {(r['method'],r['seed']) for r in d['runs']}==MATRIX,'selection matrix')
    return d

def load_selected(path,row):
    raw=Path(path).read_bytes()
    require(len(raw)==row['size_bytes'] and hashlib.sha256(raw).hexdigest()==row['checkpoint_sha256'],'checkpoint bytes')
    p=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    require(p['identity']==row['identity'] and p['identity_sha256']==row['identity_sha256']==semantic_digest(p['identity']),'checkpoint identity')
    require(p['identity']['method']==row['method'] and p['identity']['seed']==row['seed'] and p['epoch']==39,'checkpoint run')
    require(p['best_epoch']==row['best_epoch'] and p['best_model_digest']==row['best_model_digest'],'selected best')
    require(p['best_rows']==row['validation_rows'],'frozen validation predictions')
    require(state_digest(p['best_model_state'])==row['best_model_digest'],'best tensors')
    require(all(torch.isfinite(v).all() for v in p['best_model_state'].values()),'finite best tensors')
    return p['best_model_state']

def expected_population(view, *, route='B'):
    require(route in ('A','B') and view.route==route and view.role=='target' and view.tasks==PRIMARY and view.split in ('validation','test'),'prediction scope')
    return {(t,view.sample_ids[i]):dict(label=float(view.labels[i,j]),canonical=view.canonical[i],group=view.groups[i])
            for j,t in enumerate(PRIMARY) for i in range(len(view.sample_ids)) if np.isfinite(view.labels[i,j])}

def metrics(rows,view, *, route='B'):
    expected=expected_population(view,route=route);keys=[(r['task'],r['sample_id']) for r in rows]
    require(len(keys)==len(set(keys)) and set(keys)==set(expected),'prediction population differs')
    for r in rows:
        require(r['split']==view.split and all(r[k]==v for k,v in expected[(r['task'],r['sample_id'])].items()),'prediction identity differs')
        require(all(type(r[k]) in (int,float) and math.isfinite(r[k]) for k in ('prediction','lower','upper','label')),'nonfinite or nonnumeric predictions')
        require(r['lower']<=r['prediction']<=r['upper'],'quantile order')
    out={}
    for t in PRIMARY:
        e=np.asarray([r['prediction']-r['label'] for r in rows if r['task']==t],dtype=np.float64)
        require(len(e)>0,'empty endpoint')
        out[t]=dict(n=len(e),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))))
    return dict(endpoints=out,macro_rmse=float(np.mean([v['rmse'] for v in out.values()])),
                macro_mae=float(np.mean([v['mae'] for v in out.values()])),interval_status='UNCALIBRATED_NOT_COVERAGE_CLAIM')

def compare_validation(actual,expected):
    def keyed(rows):
        d={(r['task'],r['sample_id']):r for r in rows}
        require(len(d)==len(rows),'duplicate validation row');return d
    a,b=keyed(actual),keyed(expected);require(set(a)==set(b),'validation population')
    errors=[]
    for k,r in a.items():
        ref=b[k]
        require(all(r[f]==ref[f] for f in ('task','sample_id','split','label','canonical','group')),'validation identity')
        require(math.isfinite(r['prediction']) and math.isfinite(ref['prediction']),'validation nonfinite')
        require(math.isclose(r['prediction'],ref['prediction'],abs_tol=1e-6,rel_tol=1e-6),'validation regression')
        errors.append(abs(r['prediction']-ref['prediction']))
    return dict(n=len(errors),max_abs_error=max(errors),atol=1e-6,rtol=1e-6,status='PASS')

def predict(state,row,view,device,cache):
    from dataset import DataCollator
    # Random initialization is construction only, fully replaced by selected
    # checkpoint tensors; B0 factory does not relabel the method in provenance.
    model=build_human5_model(SimpleNamespace(**row['identity']['architecture']),method='B0',seed=row['seed'])
    model.load_state_dict(state,strict=True);model.to(device);model.eval();records=[]
    scaler=row['identity']['scaler']['scaler']
    require(scaler['task_names']==list(PRIMARY),'scaler order')
    with torch.no_grad():
        for j,t in enumerate(PRIMARY):
            ds=GraphTaskView(view,t)
            for start in range(0,len(ds),32):
                ids=list(range(start,min(start+32,len(ds))));graphs=[]
                for i in ids:
                    key=(view.split,t,i)
                    if key not in cache:cache[key]=ds[i]
                    graphs.append(cache[key])
                batch=DataCollator()(graphs)
                require(not batch.is_empty and batch.y.numel()==len(ids),'graph dropped')
                raw=model(batch.to(device),task_name=t)[t]
                require(tuple(raw.shape)==(len(ids),3) and torch.isfinite(raw).all(),'prediction channels')
                decoded=decode_prediction(raw,'quantile')
                values={k:(v.double()*scaler['stds'][j]+scaler['means'][j]).cpu().reshape(-1).tolist() for k,v in decoded.as_dict().items()}
                for n,i in enumerate(ids):
                    idx=ds.indices[i]
                    records.append(dict(task=t,split=view.split,sample_id=ds.get_sample_id(i),canonical=view.canonical[idx],
                        group=view.groups[idx],label=float(view.labels[idx,j]),prediction=values['median'][n],
                        lower=values['lower'][n],upper=values['upper'][n]))
    del model
    return records

def write_json(path,value):
    raw=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    with Path(path).open('x',encoding='utf8') as f:f.write(raw)

def export_all(lock,table,output,*,device,checkpoint_resolver=None,predictor=predict,
               route='B',task_id='S5C_ROUTEB_15_FORMAL_TEST_20260915',lock_sha=LOCK_SHA):
    output=Path(output);require(not output.exists(),'output exists')
    require(len(lock['runs'])==15 and {(r['method'],r['seed']) for r in lock['runs']}==MATRIX,'matrix')
    require(route in ('A','B'),'export route')
    train=table.view(route,'target','train');validation=table.view(route,'target','validation')
    test=table.view(route,'target','test');sc=TrainOnlyScaler.fit(train).to_dict()
    require(all(r['identity']['input_identity']==table.identity for r in lock['runs']),'data identity')
    for r in lock['runs']:require(r['identity']['scaler']==sc,'scaler identity')
    output.mkdir(parents=True,exist_ok=False);cache={};states={};regressions=[]
    # All fifteen gates complete before the first test forward. No run can
    # inherit another run's receipt or use a failed/partial previous directory.
    for r in lock['runs']:
        key=f"{r['method']}_seed{r['seed']}"
        path=r['checkpoint_path'] if checkpoint_resolver is None else checkpoint_resolver(r)
        state=load_selected(path,r);states[key]=state
        rows=predictor(state,r,validation,device,cache);metrics(rows,validation,route=route)
        check=compare_validation(rows,r['validation_rows']);check.update(method=r['method'],seed=r['seed'])
        write_json(output/(key+'_validation.json'),rows);regressions.append(check)
    write_json(output/'validation_regression.json',regressions)
    result=[]
    for r in lock['runs']:
        key=f"{r['method']}_seed{r['seed']}"
        rows=predictor(states[key],r,test,device,cache);m=metrics(rows,test,route=route)
        target=output/(key+'_test.json');write_json(target,rows)
        # Parse the actual saved artifact and recalculate, not a PASS field.
        need=metrics(strict_json(target.read_bytes()),test,route=route);require(need==m,'saved metric differs')
        result.append(dict(method=r['method'],seed=r['seed'],best_epoch=r['best_epoch'],
            checkpoint_sha256=r['checkpoint_sha256'],best_model_digest=r['best_model_digest'],
            prediction_file=target.name,prediction_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),metrics=m))
    summary=dict(task_id=task_id,selection_lock_sha256=lock_sha,
        input_identity=table.identity,runs=result,validation_regressions=regressions,
        training_performed=False,calibration_predictions_accessed=False,acceptance_status='PENDING_REVIEW')
    write_json(output/'test_summary.json',summary)
    return summary
