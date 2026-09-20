"""End-to-end orchestration with synthetic CPU data; no real GPU/data access."""
from types import SimpleNamespace
import hashlib
import json
import torch
from tests.test_p2_engine import View, Model
from tests.test_p2_execution import release


def test_smoke_orchestration_cpu_double(tmp_path,monkeypatch):
    import scripts.run_p2_smoke as runner
    import p2_data,p2_engine,p2_model
    class Data:
        def __init__(self,*args):pass
        def source(self,*args):
            v=View(True)
            v.rows['train']['source']=[dict(sample_id=str(i),canonical=str(i),group=str(i),label=float(i%7)) for i in range(700)]
            v.scalers={'source':p2_data.fit_scaler(v.rows['train']['source'])};v.identity=p2_data.digest(v.rows)
            return v
        def target(self):return View()
        def close(self):pass
    def build(role,seed,source_encoder=None):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            m=Model(('source',) if role=='source' else ('a','b','c'))
        if source_encoder is not None:m.encoder.load_state_dict(source_encoder,strict=True)
        return m
    original=p2_engine.P2Engine
    def cpu_engine(*args,**kwargs):
        kwargs['device']='cpu'
        e=original(*args,**kwargs)
        # Emulate serialized server identity while all computation stays CPU.
        e.identity['device']='cuda:0';e.identity_sha=p2_data.digest(e.identity)
        return e
    monkeypatch.setattr(p2_data,'P2Data',Data)
    monkeypatch.setattr(p2_model,'build_model',build)
    monkeypatch.setattr(p2_engine,'P2Engine',cpu_engine)
    monkeypatch.setattr(runner,'sys',SimpleNamespace(platform='linux',version='synthetic'))
    monkeypatch.setattr(runner,'load_release',lambda *args:release())
    monkeypatch.setattr(runner,'load_contract',lambda *args:(None,None))
    def git(repo,*args):
        if args==('rev-parse','HEAD'):return 'a'*40
        if args==('rev-parse','--git-common-dir'):return str(tmp_path)
        return ''
    monkeypatch.setattr(runner,'git',git)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for name,value in [('is_available',True),('device_count',1),('get_device_name','SYNTHETIC A6000'),
                       ('is_current_stream_capturing',False),('max_memory_allocated',0),('empty_cache',None),('reset_peak_memory_stats',None)]:
        monkeypatch.setattr(torch.cuda,name,lambda *args,v=value:v)
    args=['run_p2_smoke']
    for key,value in dict(authorization='x',authorization_sha='b'*64,approval='x',protocol='x',members='x',datastore='x',output=str(tmp_path/'output')).items():
        args.extend(['--'+key.replace('_','-'),value])
    import sys
    monkeypatch.setattr(sys,'argv',args)
    deterministic=torch.are_deterministic_algorithms_enabled()
    try:runner.main()
    finally:torch.use_deterministic_algorithms(deterministic)
    root=tmp_path/'output'
    summary=json.loads((root/'summary.json').read_text())
    assert summary['updates']==112 and len(summary['runs'])==7
    assert summary['runs'][-1]['resume']['updates']==30
    assert len(list(root.glob('*/epoch_*.json')))==18
    checksums=json.loads((root/'checksums.json').read_text())
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==sha for p,sha in checksums.items())
    import scripts.verify_p2_smoke as verifier
    monkeypatch.setattr(verifier,'build_model',build)
    monkeypatch.setattr(verifier,'P2Engine',cpu_engine)
    assert verifier.verify(root,release(),'b'*64,Data())['content_status']=='PASS'
    # Re-hashing a scientifically wrong export must not make it valid.
    name='P2SMOKE_FROZEN/epoch_00.json'
    path=root/name;value=json.loads(path.read_text());value['validation_rows'][0]['label']+=1
    path.write_text(json.dumps(value),encoding='utf8')
    checksums[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'checksums.json').write_text(json.dumps(checksums),encoding='utf8')
    import pytest
    from p2_contract import P2Error
    with pytest.raises(P2Error):verifier.verify(root,release(),'b'*64,Data())
    def failing_engine(*args,**kwargs):
        e=cpu_engine(*args,**kwargs)
        def fail():raise RuntimeError('synthetic step failure')
        e.step=fail
        return e
    monkeypatch.setattr(p2_engine,'P2Engine',failing_engine)
    args[-1]=str(tmp_path/'failed_output')
    with pytest.raises(RuntimeError,match='synthetic step failure'):runner.main()
    failure=json.loads((tmp_path/'failed_output'/'failure.json').read_text())
    assert failure['attempted_updates']==1 and failure['completed_runs']==0
