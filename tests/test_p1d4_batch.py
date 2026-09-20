from copy import deepcopy
import pytest
from p1d4_batch import execute_matrix,claim,REGISTRY
from p1d4_identity import contract_for
from p1d4_reuse import bound_history
from p1d_optimization import OptimizationError


def fake_results(jobs,*,low):
    results={}
    for j in jobs:
        reference=bound_history(f"{j['setting']}_B1_s42_existing")['history']
        history=deepcopy(reference)
        for row in history:
            for ep in row['endpoints'].values():ep['rmse']=1. if j['arm'].endswith('_low')==low else 2.
        results[j['run_id']]=dict(job=j,result=dict(validation_status='PASS',history=history))
    return results


@pytest.mark.parametrize('low',[True,False])
def test_complete_matrix_no_midrun_review_and_exact_budget(low):
    phases=[]
    def run(jobs):
        phases.append(jobs)
        return fake_results(jobs,low=low)
    result=execute_matrix(run)
    assert len(phases)==2 and len(phases[0])==12 and len(phases[1])==24
    assert len(result['results'])==36 and result['acceptance_status']=='PENDING_REVIEW'
    assert result['plan']['budget']['model_epochs']<=1280
    assert result['plan']['budget']['optimizer_updates']<=12960
    assert sum(j['action']=='TRAIN' for j in phases[0])==8


def test_failed_screen_never_starts_replication():
    calls=[]
    def run(jobs):
        calls.append(jobs);result=fake_results(jobs,low=True)
        del result[next(iter(result))]
        return result
    with pytest.raises(OptimizationError,match='incomplete screening'):execute_matrix(run)
    assert len(calls)==1


def test_invalid_screen_cannot_unlock_by_pass_status():
    def run(jobs):
        result=fake_results(jobs,low=True)
        result[next(iter(result))]['result']['history'][0]['endpoints']={}
        return result
    with pytest.raises(OptimizationError):execute_matrix(run)


def test_persistent_claim_not_refunded_by_new_output(tmp_path):
    from p1d_schedule import screening_jobs
    j=screening_jobs()[1]
    claim(tmp_path,j,tmp_path/'one','a'*40)
    with pytest.raises(FileExistsError):claim(tmp_path,j,tmp_path/'two','a'*40)
    assert (tmp_path/REGISTRY/(j['run_id']+'.json')).is_file()


def test_changed_job_budget_refused_before_claim(tmp_path):
    from p1d_schedule import screening_jobs
    j=screening_jobs()[1];j['new_epochs']=80
    with pytest.raises(OptimizationError):claim(tmp_path,j,tmp_path/'one','a'*40)
    assert not (tmp_path/REGISTRY).exists()


def test_gpu_filter_does_not_steal_occupied_slots(monkeypatch):
    import p1d4_batch
    answers=iter(['0, GPU-zero\n1, GPU-one\n2, GPU-two\n3, GPU-three\n','GPU-one, 100\nGPU-three, 101\n'])
    monkeypatch.setattr(p1d4_batch.subprocess,'check_output',lambda *a,**kw:next(answers))
    assert p1d4_batch.free_gpus([0,1,2,3])==[0,2]


def test_child_failure_retains_real_exit_and_does_not_read_pass(tmp_path,monkeypatch):
    import p1d4_batch
    from p1d_schedule import screening_jobs
    j=screening_jobs()[1]
    monkeypatch.setattr(p1d4_batch,'check_commit',lambda *a:None)
    monkeypatch.setattr(p1d4_batch,'free_gpus',lambda x:x)
    class Failed:
        pid=123
        def __init__(self,argv,**kw):
            assert kw['env']['CUDA_VISIBLE_DEVICES']=='2'
            kw['stderr'].write(b'real failure')
        def wait(self):return 7
    monkeypatch.setattr(p1d4_batch.subprocess,'Popen',Failed)
    with pytest.raises(OptimizationError,match='worker failed'):
        p1d4_batch.run_child(tmp_path,tmp_path,j,'a'*40,2,tmp_path/'split',tmp_path/'source')
    log=tmp_path/'logs'/j['run_id']
    assert p1d4_batch.read(log/'exit.json')['exit_code']==7
    assert (log/'stderr.log').read_bytes()==b'real failure'


def test_supervisor_runs_both_phases_and_archives_all_claims(tmp_path,monkeypatch):
    import p1d4_batch as module
    repo=tmp_path/'repo';repo.mkdir()
    root=tmp_path/'batch';root.mkdir()
    (repo/REGISTRY).mkdir(parents=True)
    module.write(repo/REGISTRY/'batch.json',dict(root=str(root.resolve())))
    module.write(root/'launch.json',dict(commit='a'*40,task_id=module.TASK))
    monkeypatch.setattr(module,'check_commit',lambda *args:None)
    monkeypatch.setattr(module,'free_gpus',lambda allowed:allowed)
    monkeypatch.setattr(module.time,'sleep',lambda duration:None)
    monkeypatch.setattr(module,'run_child',lambda repo,root,j,*args:fake_results([j],low=True)[j['run_id']])
    assert module.run_batch(repo,root,'a'*40,[0,1,2,3],tmp_path/'split',tmp_path/'source')==0
    assert len(list((root/'attempt_claims').glob('*_s*.json')))==36
    summary=module.read(root/'batch_summary.json')
    assert len(summary['results'])==36 and summary['validation_status']=='PASS'
    assert (root/'resolved_plan.json').is_file()
    for entry in (root/'attempt_claims').glob('*_s*.json'):
        record=module.read(entry)
        assert (record['selection_sha256'] is not None)==(record['job']['phase']=='replication')
