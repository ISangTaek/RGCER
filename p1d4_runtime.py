"""Fixed P1D4 target engine wiring. The batch layer owns attempts and budgets."""
from pathlib import Path
import json,math

from p1d4_identity import LOCK,contract_for
from p1d4_reuse import import_alias
from p1d_optimization import require
from p1d_tox import ToxFactory,file_sha
from p1d_tox_training import ToxRunAdapter
from p1d_routes import RouteFactory,SingleRunAdapter
from p1d_smoke import _read_json


def route_history(output):
    output=Path(output);summary=_read_json(output/'training_summary.json');history=[]
    for record in summary['history']:
        epoch=record['epoch'];rows=_read_json(output/f'validation_epoch_{epoch:03d}.json')
        endpoints={t:dict(values) for t,values in record['validation']['endpoints'].items()}
        for task,values in endpoints.items():
            selected=[r for r in rows if r['task']==task]
            require(len(selected)==values['n'] and selected,'route metric counts')
            mean=math.fsum(r['label'] for r in selected)/len(selected)
            sst=math.fsum((r['label']-mean)**2 for r in selected)
            sse=math.fsum((r['prediction']-r['label'])**2 for r in selected)
            values.update(r2=None if len(selected)<2 or sst==0 else 1-sse/sst,
                          r2_reason='insufficient_or_constant_labels' if len(selected)<2 or sst==0 else None)
        history.append(dict(epoch=epoch,split='validation',endpoints=endpoints))
    return history


def factory_for(repo,*,setting,seed,split_manifest,source_lock,device):
    from dataset115_adapter import Dataset115Table
    from toxacute_datastore import ToxAcuteDataStore
    repo=Path(repo).resolve();lock=json.loads(LOCK.read_bytes());inputs=lock['inputs']
    contract=contract_for(setting,seed)
    store=ToxAcuteDataStore.resolve(repo/inputs['datastore_relative'])
    require(file_sha(store.root/'split_manifest.json')==inputs['tox_manifest_sha256'],'frozen DataStore bytes')
    if setting=='ToxAcute':
        return ToxFactory(repo=repo,datastore=store.root,contract=contract,device=device)
    require(file_sha(repo/inputs['csv_relative'])==inputs['csv_sha256'],'CSV bytes')
    require(file_sha(split_manifest)==inputs['split_sha256'],'split bytes')
    require(file_sha(source_lock)==inputs['source_lock_sha256'],'source lock bytes')
    table=Dataset115Table.load(repo/inputs['csv_relative'],split_manifest,store.root/'split_manifest.json',
                               expected_tox_sha=inputs['tox_manifest_sha256'])
    if setting=='B':extra=dict(source_repo=repo,source_lock=source_lock)
    else:
        path=(repo/lock['route_a_sources'][str(seed)]).resolve()
        require(path.is_relative_to(repo),'source path outside repository')
        extra=dict(source_output=path)
    return RouteFactory(table,route=setting,seed=seed,expected_identity=contract,**extra)


def execute_job(repo,job,output,*,split_manifest,source_lock,device):
    output=Path(output)
    require(not output.exists(),'existing job output')
    if job['action']=='REUSE' and job['setting']!='ToxAcute':
        # Accepted A/B epochs and replays are inherited by exact file identity.
        result=import_alias(repo,job['alias'])
    else:
        factory=factory_for(repo,setting=job['setting'],seed=job['seed'],split_manifest=split_manifest,
                            source_lock=source_lock,device=device)
        if job['action']=='REUSE':result=import_alias(repo,job['alias'],tox_factory=factory)
        elif job['setting']=='ToxAcute':result=ToxRunAdapter(factory,job['arm']).run(output)
        else:
            verified=SingleRunAdapter(factory,arm=job['arm'],device=device).run(output)
            result=dict(verification=verified,history=route_history(output),
                        validation_status='PASS',acceptance_status='PENDING_REVIEW')
    return dict(job=job,result=result)


def verify_job(repo,job,output,*,split_manifest,source_lock,device):
    """Independent read-only entry; it cannot create an optimizer update."""
    if job['action']=='REUSE':
        factory=(factory_for(repo,setting='ToxAcute',seed=job['seed'],split_manifest=split_manifest,
                             source_lock=source_lock,device=device) if job['setting']=='ToxAcute' else None)
        return dict(job=job,result=import_alias(repo,job['alias'],tox_factory=factory))
    factory=factory_for(repo,setting=job['setting'],seed=job['seed'],split_manifest=split_manifest,
                        source_lock=source_lock,device=device)
    if job['setting']=='ToxAcute':result=ToxRunAdapter(factory,job['arm']).verify(output)
    else:
        verified=SingleRunAdapter(factory,arm=job['arm'],device=device).verify(output)
        result=dict(verification=verified,history=route_history(output),
                    validation_status='PASS',acceptance_status='PENDING_REVIEW')
    return dict(job=job,result=result)
