"""P1D5 fixed-checkpoint inference. No Trainer, optimizer, fit, or selection."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import hashlib,math
import torch
from p1d_optimization import require
from p1d_routes import _exact
from p1d_tox import TASKS,file_sha,load_legacy
from p1d_smoke import model_digest
from dataset115_smoke import state_digest
from dataset115_test_export import strict_json,write_json,predict as route_predict

LOCK_SHA='13f0038b4bf0cc5799abc20878a525e80c66bd53af6e0e259269702668c17a5c'
TASK='P1D5_TEST_20260920'

def load_lock(path):
    raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==LOCK_SHA,'frozen lock SHA')
    lock=strict_json(raw);validate_lock(lock);return lock

def validate_lock(lock):
    require(lock['schema']=='p1d5_test_v1' and lock['task_id']==TASK and lock['training_performed'] is False,'lock scope')
    rows=lock['rows'];matrix={(s,f,n) for s in ('ToxAcute','A','B') for f in ('B1','HF') for n in range(42,47)}
    require(len(rows)==30 and {(r['setting'],r['family'],r['seed']) for r in rows}==matrix,'fixed matrix')
    require(len({r['run_id'] for r in rows})==30,'unique run identity')
    for r in rows:
        arm='B1_high' if r['setting']=='B' and r['family']=='B1' else r['family']+'_low'
        require(type(r['seed']) is int and type(r['best_epoch']) is int and 0<=r['best_epoch']<40,'seed/epoch type')
        require(r['arm']==arm and r['run_id']==f"{r['setting']}_{arm}_s{r['seed']}",'selected arm')
        require(r['action']==('REUSE' if r['setting']=='B' and r['family']=='B1' else 'EXPORT'),'action')
    require(lock['validation_atol']==1e-6 and lock['validation_rtol']==1e-6,'fixed replay tolerance')

def selected_path(row,repo,training_root):
    cp=row['checkpoint']
    base=Path(training_root if 'canonical_member' in cp else repo).resolve()
    p=(base/cp.get('canonical_member',cp.get('server_repo_relative_path'))).resolve()
    require(p.is_relative_to(base) and p.is_file(),'missing or outside-root checkpoint')
    return p

def load_state(path,row):
    cp=row['checkpoint'];require(file_sha(path)==cp['sha256'],'selected checkpoint SHA')
    if 'size_bytes' in cp:require(Path(path).stat().st_size==cp['size_bytes'],'checkpoint size')
    p=load_legacy(path,cp['sha256']) if row['checkpoint_kind']=='tox_legacy' else torch.load(path,map_location='cpu',weights_only=True)
    require(type(p['epoch']) is int and p['epoch']==row['best_epoch'],'checkpoint epoch')
    state=p['model_state'];require(all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in state.values()),'finite weights')
    digest=model_digest(state) if row['setting']=='ToxAcute' else state_digest(state)
    require(digest==row['model_digest'],'selected tensor digest')
    if row['checkpoint_kind']=='tox_legacy':
        _exact(p['task_scalers'],row['scalers'],'legacy scaler')
        require(p['task_names']==list(TASKS),'legacy tasks')
        require(p['reproducibility']['initial_model_sha256']==row['contract']['initial_full_model_digest'],'legacy initial identity')
    else:
        _exact(p['identity'],row['identity'],'checkpoint identity')
        if row['checkpoint_kind']=='tox_new':_exact(p['scalers'],row['scalers'],'scaler')
    require(file_sha(path)==cp['sha256'],'checkpoint changed during load')
    return state

def keyed(rows):
    require(type(rows) is list and bool(rows),'nonempty records')
    result={}
    for r in rows:
        require(type(r) is dict and type(r.get('task')) is str and type(r.get('sample_id')) is str,'record ID types')
        k=r['task'],r['sample_id'];require(k not in result,'duplicate record');result[k]=r
    return result

def metrics(rows,expected,tasks,split):
    actual=keyed(rows);reference=keyed(expected)
    require(set(actual)==set(reference),'exact raw population')
    require({r['task'] for r in rows}==set(tasks),'all tasks required')
    for k,r in actual.items():
        require(r['split']==split,'split')
        for field,value in reference[k].items():
            if field not in ('prediction','lower','upper'):_exact(r.get(field),value,'raw '+field)
        require(all(type(r.get(f)) in (int,float) and math.isfinite(r[f]) for f in ('label','prediction')),'finite numeric prediction/label')
        if 'lower' in r or 'upper' in r:
            require(all(type(r.get(f)) in (int,float) and math.isfinite(r[f]) for f in ('lower','upper')),'finite quantiles')
            require(r['lower']<=r['prediction']<=r['upper'],'quantile order')
    result={}
    for t in tasks:
        rr=[r for r in rows if r['task']==t];n=len(rr)
        mean=math.fsum(r['label'] for r in rr)/n
        sse=math.fsum((r['prediction']-r['label'])**2 for r in rr)
        sst=math.fsum((r['label']-mean)**2 for r in rr)
        result[t]=dict(n=n,rmse=math.sqrt(sse/n),mae=math.fsum(abs(r['prediction']-r['label']) for r in rr)/n,
            r2=1-sse/sst if n>1 and sst>0 else None,r2_reason=None if n>1 and sst>0 else 'insufficient_or_constant_labels')
    return dict(endpoints=result,macro_rmse=math.fsum(x['rmse'] for x in result.values())/len(tasks),
        macro_mae=math.fsum(x['mae'] for x in result.values())/len(tasks))

def replay(actual,reference):
    a,b=keyed(actual),keyed(reference);require(set(a)==set(b),'replay population')
    errors=[]
    for k,r in a.items():
        for field,value in b[k].items():
            if field not in ('prediction','lower','upper'):_exact(r.get(field),value,'replay '+field)
        x,y=r['prediction'],b[k]['prediction']
        require(type(x) in (int,float) and type(y) in (int,float) and math.isfinite(x) and math.isfinite(y),'replay finite')
        require(math.isclose(x,y,rel_tol=1e-6,abs_tol=1e-6),'validation replay tolerance')
        errors.append(abs(x-y))
    return dict(n=len(errors),max_abs_error=max(errors),atol=1e-6,rtol=1e-6)

class Data:
    """Only validation views initially. test views are lazy after global gate."""
    def __init__(self,repo,split_manifest,lock):
        from toxacute_datastore import ToxAcuteDataStore
        from dataset115_adapter import Dataset115Table
        repo=Path(repo);i=lock['inputs']
        self.store=ToxAcuteDataStore.resolve(repo/i['datastore_relative'])
        require(file_sha(self.store.root/'split_manifest.json')==i['tox_manifest_sha256'],'DataStore manifest')
        require(file_sha(repo/i['csv_relative'])==i['csv_sha256'] and file_sha(split_manifest)==i['split_sha256'],'CSV/split')
        c=next(r['contract'] for r in lock['rows'] if r['setting']=='ToxAcute')
        for k,v in c['data_identity'].items():_exact(self.store.metadata[k],v,'DataStore '+k)
        self.table=Dataset115Table.load(repo/i['csv_relative'],split_manifest,self.store.root/'split_manifest.json',expected_tox_sha=i['tox_manifest_sha256'])
        self.datasets={};self.views={};self.cache={}
    def tasks(self,setting):
        from dataset115_contract import PRIMARY
        return TASKS if setting=='ToxAcute' else PRIMARY
    def view(self,setting,split):
        require(split in ('validation','test'),'data split')
        key=setting,split
        if key not in self.views:self.views[key]=self.table.view(setting,'target',split)
        return self.views[key]
    def tox_ds(self,split):
        from toxacute_datastore import ToxAcuteTaskDataset
        require(split in ('validation','test'),'tox split')
        if split not in self.datasets:self.datasets[split]={t:ToxAcuteTaskDataset(self.store,t,split=split,max_nodes=512) for t in TASKS}
        return self.datasets[split]
    def population(self,setting,split):
        if setting!='ToxAcute':
            from dataset115_test_export import expected_population
            return [dict(task=t,sample_id=sid,split=split,**v) for (t,sid),v in expected_population(self.view(setting,split),route=setting).items()]
        return [dict(task=t,sample_id=ds.get_sample_id(n),split=split,label=float(ds[n].y.reshape(-1)[0])) for t,ds in self.tox_ds(split).items() for n in range(len(ds))]

def predict(state,row,data,split,device):
    if row['setting']!='ToxAcute':
        # Cache scope must include route: A/B row numbers are not identical views.
        cache=data.cache.setdefault((row['setting'],split),{})
        return route_predict(state,row,data.view(row['setting'],split),device,cache)
    from main import _build_model_components
    from architecture.prediction_heads import decode_prediction
    from dataset import DataCollator
    args=SimpleNamespace(**row['contract']['args'])
    enc,arch,heads=_build_model_components(args,list(TASKS),torch.device(device))
    model=arch(list(TASKS),enc,heads,torch.device(device),args).to(device)
    model.load_state_dict(state,strict=True);model.eval();records=[]
    collate=DataCollator(spatial_pos_max_clip=20,max_node_filter=None)
    with torch.no_grad():
        for t,ds in data.tox_ds(split).items():
            for start in range(0,len(ds),64):
                ii=list(range(start,min(start+64,len(ds))));batch=collate([ds[n] for n in ii])
                require(not batch.is_empty and batch.y.numel()==len(ii),'dropped graph')
                batch=batch.to(device);raw=model(batch,task_name=t)[t]
                require(tuple(raw.shape)==(len(ii),3) and torch.isfinite(raw).all(),'raw quantiles')
                scaler=row['scalers'][t];median=decode_prediction(raw,'quantile').median*scaler['std']+scaler['mean']
                records.extend(dict(task=t,split=split,sample_id=str(sid),label=float(y),prediction=float(p))
                    for sid,y,p in zip(batch.sample_id,batch.y.reshape(-1).cpu(),median.reshape(-1).cpu()))
    return records

def execute(lock,data,output,loader,predictor):
    """All 30 validation/identity gates before any new test forward."""
    validate_lock(lock);output=Path(output);require(not output.exists(),'fresh output required')
    output.mkdir(parents=True);regressions=[]
    for row in lock['rows']:
        state=loader(row)
        actual=row['validation_rows'] if row['action']=='REUSE' else predictor(state,row,'validation')
        metrics(actual,data.population(row['setting'],'validation'),data.tasks(row['setting']),'validation')
        check=replay(actual,row['validation_rows']);regressions.append(dict(run_id=row['run_id'],**check))
        write_json(output/(row['run_id']+'_validation.json'),actual)
        print('VALIDATED',row['run_id'],check['max_abs_error'],flush=True)
    write_json(output/'validation_regressions.json',regressions)
    results=[]
    for row in lock['rows']:
        actual=deepcopy(row['reused_test_rows']) if row['action']=='REUSE' else predictor(loader(row),row,'test')
        m=metrics(actual,data.population(row['setting'],'test'),data.tasks(row['setting']),'test')
        filename=row['run_id']+'_test.json';write_json(output/filename,actual)
        results.append(dict(run_id=row['run_id'],setting=row['setting'],family=row['family'],arm=row['arm'],seed=row['seed'],
            action=row['action'],best_epoch=row['best_epoch'],checkpoint_sha256=row['checkpoint']['sha256'],model_digest=row['model_digest'],
            prediction_file=filename,prediction_sha256=file_sha(output/filename),metrics=m))
        print('EXPORTED' if row['action']=='EXPORT' else 'INHERITED',row['run_id'],flush=True)
    summary=dict(task_id=TASK,lock_sha256=LOCK_SHA,runs=results,training_performed=False,optimizer_updates=0,
        calibration_accessed=False,acceptance_status='PENDING_CODEX_REVIEW',scope='DEVELOPMENT_TEST_NOT_INDEPENDENT_CONFIRMATION')
    write_json(output/'test_summary.json',summary);return summary

def verify(lock,data,output):
    output=Path(output);summary=strict_json((output/'test_summary.json').read_bytes())
    require(summary['task_id']==TASK and summary['lock_sha256']==LOCK_SHA and summary['training_performed'] is False
        and type(summary['optimizer_updates']) is int and summary['optimizer_updates']==0 and summary['calibration_accessed'] is False,'saved scope')
    require(len(summary['runs'])==30 and {r['run_id'] for r in summary['runs']}=={r['run_id'] for r in lock['rows']},'saved matrix')
    saved={r['run_id']:r for r in summary['runs']};regressions=[]
    for row in lock['rows']:
        rid=row['run_id'];rec=saved[rid]
        for k in ('setting','family','arm','seed','action','best_epoch','model_digest'):_exact(rec[k],row[k],'saved '+k)
        require(rec['checkpoint_sha256']==row['checkpoint']['sha256'],'saved checkpoint')
        val=strict_json((output/(rid+'_validation.json')).read_bytes())
        metrics(val,data.population(row['setting'],'validation'),data.tasks(row['setting']),'validation')
        regressions.append(dict(run_id=rid,**replay(val,row['validation_rows'])))
        require(rec['prediction_file']==rid+'_test.json','saved filename')
        raw=(output/rec['prediction_file']).read_bytes();require(hashlib.sha256(raw).hexdigest()==rec['prediction_sha256'],'saved SHA')
        rows=strict_json(raw)
        if row['action']=='REUSE':_exact(rows,row['reused_test_rows'],'unchanged inherited test')
        _exact(metrics(rows,data.population(row['setting'],'test'),data.tasks(row['setting']),'test'),rec['metrics'],'recomputed metrics')
    _exact(regressions,strict_json((output/'validation_regressions.json').read_bytes()),'regression report')
    return dict(task_id=TASK,checked_runs=30,new_test_exports=25,inherited_test_runs=5,content_status='PASS',optimizer_updates=0,acceptance_status='PENDING_CODEX_REVIEW')
