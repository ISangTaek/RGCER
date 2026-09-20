"""Read-only smoke artifact audit; no forward, training, test or calibration."""
import argparse
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2_contract import load_contract, strict_json, require
from p2_execution import load_release, write_new, audit_validation
from p2_data import P2Data, digest
from p2_model import build_model
from p2_engine import P2Engine,state_digest


def verify(root,release,release_sha,data):
    import torch
    root=Path(root).resolve()
    def read(name):return strict_json((root/name).read_bytes())
    hashes=read('checksums.json')
    allowed={'release.json','environment.json','summary.json','budget.json'}
    for run in release['runs']:
        rid=run['run_id']
        allowed.update(rid+'/'+n for n in ('scalers.json','initial.json','final.pt','summary.json'))
        if run['role']=='target':allowed.update(rid+f'/epoch_{e:02d}.json' for e in range(6))
        if run.get('method')=='HF_low':allowed.add(rid+'/before_unfreeze.pt')
    require(set(hashes)==allowed,'delivery allowlist')
    require(not any(p.is_symlink() for p in root.rglob('*')),'delivery symlink')
    actual={str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*') if p.is_file() and p.name!='checksums.json'}
    require(set(hashes)==actual,'delivery member set')
    for name,sha in hashes.items():
        path=root/name
        require(path.resolve().is_relative_to(root) and not path.is_symlink(),'delivery path')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==sha,'delivery checksum')
    require(read('release.json')==release,'delivery release')
    env=read('environment.json');require(env['commit']==release['commit'] and env['deterministic'] is True and env['tf32'] is False,'runtime identity')
    summary=read('summary.json')
    require(summary['updates']==112 and summary['test_access'] is False and summary['calibration_access'] is False,'summary permissions/budget')
    require(len(summary['runs'])==7,'summary run count')
    counts={};sources={};audits=[]
    for run,reported in zip(release['runs'],summary['runs']):
        rid=run['run_id'];is_source=run['role']=='source';folder=root/rid
        require(reported==read(rid+'/summary.json') and reported['run']==run,'run summary identity')
        counts[rid]=run['updates']
        source_sha=None
        if is_source:
            view=data.source(run['panel'],run['route']);model=build_model('source',42)
        else:
            view=data.target();source=sources[run['source_run_id']];source_sha=source['sha']
            encoder={k.removeprefix('encoder.'):v for k,v in source['state'].items() if k.startswith('encoder.')}
            model=build_model('target',42,encoder)
        identity=dict(**run,release_sha=release_sha,commit=release['commit'],source_checkpoint_sha=source_sha)
        engine=P2Engine(model,view,method='SOURCE' if is_source else run['method'],seed=42,run_identity=identity)
        require(read(rid+'/scalers.json')==view.scalers,'train-only scalers')
        require(read(rid+'/initial.json')==dict(model_sha=engine.identity['initial_model'],view_sha=view.identity,source_checkpoint_sha=source_sha),'initial model identity')
        final=folder/'final.pt';sha=hashlib.sha256(final.read_bytes()).hexdigest()
        require(sha==reported['final_sha'],'final byte identity')
        payload=torch.load(final,map_location='cpu',weights_only=True)
        expected=dict(engine.identity,device='cuda:0',torch_version=env['torch'])
        require(payload['identity']==expected and payload['identity_sha']==digest(expected),'checkpoint identity')
        require(payload['schema']=='p2_checkpoint_v1' and payload['total_updates']==run['updates'],'checkpoint budget')
        require(payload['epoch']==(0 if is_source else 6) and payload['offset']==(1 if is_source else 0),'checkpoint cursor')
        require(state_digest(payload['model_state'])==payload['model_digest']==reported['final_model_sha'],'model digest')
        engine.model.load_state_dict(payload['model_state'],strict=True)
        engine.control.restore(payload['optimization'],allow_partial_epoch=is_source)
        for key in ('epoch','offset','total_updates','history','running'):setattr(engine,key,payload[key])
        engine.check_cursor_and_optimizer()
        for h in engine.history:require(read(rid+f"/epoch_{h['epoch']:02d}.json")==h,'epoch export identity')
        if is_source:
            require(payload['best_epoch'] is None and payload['best_state'] is None,'source has no selected best')
            sources[rid]=dict(sha=sha,state=payload['model_state'])
            audits.append(dict(run_id=rid,status='PASS',updates=1))
        else:
            check=audit_validation(engine.history,view,payload['best_epoch'])
            require(check==reported['validation'],'reported validation audit')
            require(payload['best_rows']==engine.history[payload['best_epoch']]['validation_rows'],'best prediction selection')
            require(state_digest(payload['best_state'])==payload['best_digest'],'best tensor identity')
            require(set(payload['best_state'])==set(payload['model_state']),'best tensor keys')
            require(all(torch.isfinite(v).all().item() and v.shape==payload['model_state'][k].shape for k,v in payload['best_state'].items()),'best tensor shape/finite')
            audits.append(dict(run_id=rid,**check,updates=36))
        del engine,model
    require(read('budget.json')==dict(attempted_updates=112,completed_updates=112,counts=counts),'budget tape')
    return dict(content_status='PASS',acceptance_status='PENDING_CODEX_REVIEW',scope='read_only_no_forward',updates=112,runs=audits)


def main():
    p=argparse.ArgumentParser()
    for name in ('authorization','authorization-sha','approval','protocol','members','datastore','results','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();release=load_release(a.authorization,a.authorization_sha,a.approval)
    _,members=load_contract(a.protocol,a.members);data=P2Data(members,a.datastore)
    try:write_new(a.output,verify(a.results,release,a.authorization_sha,data))
    finally:data.close()
    print('CONTENT_PASS_PENDING_CODEX_REVIEW')


if __name__=='__main__':main()
