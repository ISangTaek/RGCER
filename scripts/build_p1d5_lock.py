"""Codex-only deterministic preparation from accepted local archives; stdout JSON."""
import json,sys,hashlib
from pathlib import Path
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from p1d_tox import load_legacy,file_sha
from p1d_smoke import model_digest
from dataset115_smoke import state_digest
from p1d4_identity import contract_for
def read(p):return json.loads(p.read_bytes())
def main():
    results=REPO/'Plans&Results/Results';root=results/'065_review_20260920/canonical'
    matrix=read(root.parent/'p1d5_candidate_checkpoint_matrix.json');summary=read(root/'batch_summary.json')
    oldroot=results/'053_review_20260916/extracted/route_b_test';oldsummary=read(oldroot/'test_summary.json')
    lock=dict(schema='p1d5_test_v1',task_id='P1D5_TEST_20260920',training_performed=False,
              source_canonical_sha256=matrix['source_canonical_sha256'],inputs=read(REPO/'configs/p1d4_identity_lock.json')['inputs'],
              validation_atol=1e-6,validation_rtol=1e-6,rows=[])
    for item in matrix['rows']:
        r=dict(item);cp=item['checkpoint'];rid=item['run_id'];setting=item['setting'];seed=item['seed']
        r['contract']=contract_for(setting,seed)
        r['action']='REUSE' if setting=='B' and item['family']=='B1' else 'EXPORT'
        r.pop('planned_action')
        if 'canonical_member' in cp:
            path=root/cp['canonical_member'];assert file_sha(path)==cp['sha256']
            p=torch.load(path,map_location='cpu',weights_only=True)
            r['checkpoint_kind']='tox_new' if setting=='ToxAcute' else 'route'
            r['identity']=p['identity'];r['model_digest']=model_digest(p['model_state']) if setting=='ToxAcute' else state_digest(p['model_state'])
            r['scalers']=p['scalers'] if setting=='ToxAcute' else p['identity']['scaler']
            name=f"epoch_{item['best_epoch']:03d}_validation.json" if setting=='ToxAcute' else f"validation_epoch_{item['best_epoch']:03d}.json"
            r['validation_rows']=read(root/'runs'/rid/name)
        elif setting=='ToxAcute':
            path=results/'063_review_20260920/extracted/collection/files'/cp['alias']/Path(cp['server_repo_relative_path']).name
            p=load_legacy(path,cp['sha256']);r['checkpoint_kind']='tox_legacy'
            r['model_digest']=model_digest(p['model_state']);r['scalers']=p['task_scalers']
            r['validation_rows']=summary['results'][rid]['result']['validation_replay']['rows']
        else:
            ref=next(x for x in oldsummary['runs'] if x['method']=='B1' and x['seed']==seed)
            oldtrain=results/f'052_review_20260915/extracted/runs/B1_seed{seed}'
            assert read(oldtrain/'training_summary.json')['best_epoch']==ref['best_epoch']==item['best_epoch']
            assert read(oldtrain/f"epoch_{item['best_epoch']:03d}.receipt.json")['sha256']==cp['sha256']
            r['checkpoint_kind']='route';r['identity']=read(oldtrain/'resolved_config.json')
            r['model_digest']=ref['best_model_digest'];r['scalers']=r['identity']['scaler']
            r['validation_rows']=read(oldroot/f'B1_seed{seed}_validation.json')
            raw=(oldroot/ref['prediction_file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['prediction_sha256']
            r['reused_test_rows']=json.loads(raw)
            r['reuse_provenance']=dict(source_review='053_review_20260916',member='route_b_test/'+ref['prediction_file'],
                prediction_sha256=ref['prediction_sha256'],source_checkpoint_sha256=ref['checkpoint_sha256'],
                same_best_tensor_digest=ref['best_model_digest'])
        lock['rows'].append(r)
    raw=json.dumps(lock,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if len(sys.argv)==3 and sys.argv[1]=='--output':
        with Path(sys.argv[2]).open('x',encoding='utf8') as f:f.write(raw)
        print(file_sha(sys.argv[2]))
    else: print(raw,end='')
if __name__=='__main__':main()
