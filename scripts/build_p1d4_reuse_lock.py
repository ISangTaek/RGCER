"""Read accepted 047/052/055/063 evidence and print an inherited-history lock.

Historical JSON may contain unrelated routing NaN diagnostics. Only finite
validation endpoint metrics are inherited; original bytes remain hash-bound.
"""
import hashlib,json,math,zipfile
from pathlib import Path


def sha(raw):return hashlib.sha256(raw).hexdigest()


def build(repo):
    root=repo/'Plans&Results/Results'
    assets=json.loads((root/'2026-09-20_P1D优化诊断准备/optimization_assets.json').read_bytes())
    identities=json.loads((repo/'configs/p1d4_identity_lock.json').read_bytes())
    s3=json.loads((repo/'configs/p1d_accepted_s3_reuse.json').read_bytes())
    archive=root/'047_2026-09-15_S4E8历史轨迹与消融采集.zip'
    runs={}
    def tox_history(raw):
        source=json.loads(raw)['history']
        assert len(source)==40
        counts=identities['settings']['ToxAcute']['42']['counts']['validation']
        rows=[]
        for e,r in enumerate(source):
            assert r['epoch']==e
            v=r['validation'];assert v['selection_scope']=='human3'
            ep={t:dict(n=n,rmse=v['tasks'][t]['RMSE']) for t,n in counts.items()}
            assert all(type(x['rmse']) in (int,float) and math.isfinite(x['rmse']) and x['rmse']>=0 for x in ep.values())
            assert abs(math.fsum(x['rmse'] for x in ep.values())/3-v['human3_macro_rmse'])<1e-12
            rows.append(dict(epoch=e,split='validation',endpoints=ep))
        return rows
    def file_entry(relative,raw):return dict(path=relative,sha256=sha(raw),size_bytes=len(raw))
    with zipfile.ZipFile(archive) as z:
        for r in assets:
            if r['method']!='B1':continue
            setting,seed=r['setting'],r['seed'];alias=f'{setting}_B1_s{seed}_existing'
            best=r['best_checkpoint']
            if setting=='ToxAcute':
                raw=z.read(r['history_member']);assert sha(raw)==r['history_sha256']
                path=best['path'].split('/RGCER/',1)[1]
                hist=tox_history(raw)
                files=[file_entry(str(Path(path).parent/'metrics.json').replace('\\','/'),raw),
                       dict(path=path,sha256=best['sha256'],size_bytes=best['size_bytes'])]
                provenance=dict(archive=archive.name,member=r['history_member'],member_sha256=sha(raw))
                replay=True
            else:
                local=repo/Path(r['evidence_root'])
                server=('.tmp/s5f2_work_20260916T012948Z/route_a_20/runs/' if setting=='A' else
                        '.tmp/s5b_work_20260915T141617Z/route_b_15/runs/')+f'B1_seed{seed}'
                raw=(local/'training_summary.json').read_bytes();summary=json.loads(raw)
                assert summary['complete'] is True and summary['completed_epochs']==40
                hist=[dict(epoch=h['epoch'],split='validation',endpoints=h['validation']['endpoints']) for h in summary['history']]
                files=[file_entry(server+'/training_summary.json',raw),
                       file_entry(server+'/resolved_config.json',(local/'resolved_config.json').read_bytes())]
                for e in range(40):
                    name=f'validation_epoch_{e:03d}.json'
                    files.append(file_entry(server+'/'+name,(local/name).read_bytes()))
                files.append(dict(path=server+'/'+best['path'],sha256=best['sha256'],size_bytes=best['size_bytes']))
                provenance=dict(review_root=r['evidence_root'],summary_sha256=sha(raw))
                path=server+'/'+best['path'];replay=False
            winner=min(range(40),key=lambda e:math.fsum(x['rmse'] for x in hist[e]['endpoints'].values())/len(hist[e]['endpoints']))
            runs[alias]=dict(setting=setting,seed=seed,arm='B1_high',history=hist,files=files,
                            best_path=path,best_sha256=best['sha256'],best_epoch=winner,
                            best_rmse=math.fsum(x['rmse'] for x in hist[winner]['endpoints'].values())/len(hist[winner]['endpoints']),
                            fresh_validation_replay=replay,provenance=provenance)
    for r in s3['runs']:
        local=root/f"063_review_20260920/extracted/collection/files/{r['run_id']}"
        raw=(local/'metrics.json').read_bytes();hist=tox_history(raw)
        name=f"graphormer_d7_s3_e40_seed{r['seed']}_best.pt"
        files=[file_entry(r['relative_path']+'/metrics.json',raw)]
        for n,h in r['weight_sha256'].items():
            original=local/n
            assert sha(original.read_bytes())==h
            files.append(dict(path=r['relative_path']+'/'+n,sha256=h,size_bytes=original.stat().st_size))
        runs[r['run_id']]=dict(setting='ToxAcute',seed=r['seed'],arm='HF_low',history=hist,files=files,
            best_path=r['relative_path']+'/'+name,best_sha256=r['weight_sha256'][name],
            best_epoch=r['best_epoch'],best_rmse=r['best_validation_macro_rmse'],fresh_validation_replay=True,
            provenance=dict(archive_sha256=s3['source_archive_sha256'],member_sha256=sha(raw)))
    assert len(runs)==18
    return dict(schema='p1d4_reuse_lock_v1',execution_authorized=False,runs=runs)


if __name__=='__main__':
    print(json.dumps(build(Path(__file__).resolve().parents[1]),ensure_ascii=False,separators=(',',':'),allow_nan=False))
