"""S5G Route A test authorization: only the 15 accepted 055 best checkpoints."""
from pathlib import Path
import hashlib
from dataset115_training import require
from dataset115_test_export import strict_json,MATRIX,export_all as shared_export,predict,metrics,compare_validation

LOCK_SHA='008f1c2f98761a495089bdc071271c38891e30312f6b66990f536238c3aab85a'
TASK_ID='S5G_ROUTEA_15_FORMAL_TEST_20260917'
TRAINING_COMMIT='c350e58ced184861210f915ac6739acd30b289bf'


def load_lock(path):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==LOCK_SHA,'unapproved Route A selection lock')
    lock=strict_json(raw);validate_lock(lock)
    return lock


def validate_lock(lock):
    require(lock['schema']=='s5g_route_a_selected_v1' and lock['training_commit']==TRAINING_COMMIT,'Route A lock provenance')
    require(len(lock['runs'])==15 and {(r['method'],r['seed']) for r in lock['runs']}==MATRIX,'Route A selection matrix')
    for r in lock['runs']:
        identity=r['identity']
        require(type(r['seed']) is int and type(r['best_epoch']) is int and 0<=r['best_epoch']<40,'Route A selection types')
        require(identity['schema']=='dataset115_route_a_epoch_v1' and identity['method']==r['method']
            and identity['seed']==r['seed'] and identity['scaler']['route']=='A','Route A checkpoint scope')


def export_all(lock,table,output,*,device,checkpoint_resolver=None,predictor=predict):
    validate_lock(lock)
    return shared_export(lock,table,output,device=device,checkpoint_resolver=checkpoint_resolver,predictor=predictor,
        route='A',task_id=TASK_ID,lock_sha=LOCK_SHA)


def verify_saved(lock,table,output):
    """CPU re-read: raw populations, all numerical metrics, selection and hashes."""
    validate_lock(lock);output=Path(output)
    def read(name):return strict_json((output/name).read_bytes())
    summary=read('test_summary.json')
    require(summary['task_id']==TASK_ID and summary['selection_lock_sha256']==LOCK_SHA
        and summary['input_identity']==table.identity and summary['training_performed'] is False
        and summary['calibration_predictions_accessed'] is False,'saved export identity/scope')
    runs=summary['runs']
    require(len(runs)==15 and {(r['method'],r['seed']) for r in runs}==MATRIX,'saved export matrix')
    selected={(r['method'],r['seed']):r for r in lock['runs']}
    validation=table.view('A','target','validation');test=table.view('A','target','test');regressions=[]
    for row in runs:
        ref=selected[(row['method'],row['seed'])];key=f"{row['method']}_seed{row['seed']}"
        for f in ('best_epoch','checkpoint_sha256','best_model_digest'):require(row[f]==ref[f],'saved selected identity')
        v=read(key+'_validation.json');metrics(v,validation,route='A')
        check=compare_validation(v,ref['validation_rows']);check.update(method=row['method'],seed=row['seed']);regressions.append(check)
        require(row['prediction_file']==key+'_test.json','saved prediction path')
        raw=(output/row['prediction_file']).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==row['prediction_sha256'],'saved prediction SHA')
        require(metrics(strict_json(raw),test,route='A')==row['metrics'],'saved recomputed metrics')
    require(summary['validation_regressions']==regressions==read('validation_regression.json'),'saved validation regressions')
    return dict(task_id=TASK_ID,content_status='PASS',runs_checked=15,training_performed=False,
        acceptance_status='PENDING_CODEX_REVIEW',selection_lock_sha256=LOCK_SHA)
