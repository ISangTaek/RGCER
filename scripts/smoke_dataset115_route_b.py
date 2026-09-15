"""Frozen seed42 Route B one-step smoke: train-only, no formal test access."""
import argparse
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('csv','split-manifest','tox-manifest','expected-tox-file-sha256','source-lock','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',choices=['cpu','cuda:0'],required=True)
    a=parser.parse_args(); output=Path(a.output)
    if output.exists():parser.error('output exists; choose a new run path')
    try:
        import torch
        from dataset115_source import load_route_b_encoder
        from dataset115_adapter import Dataset115Table,TrainOnlyScaler,GraphTaskView
        from dataset115_contract import PRIMARY
        from dataset115_smoke import one_step,verify_step_records
        from dataset import DataCollator
        if a.device=='cuda:0' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
        repo=Path(__file__).resolve().parents[1]
        source,args,source_receipt=load_route_b_encoder(repo,a.source_lock,42)
        table=Dataset115Table.load(a.csv,a.split_manifest,a.tox_manifest,expected_tox_sha=a.expected_tox_file_sha256)
        view=table.view('B','target','train');scaler=TrainOnlyScaler.fit(view)
        batches={};ids={}
        for t in PRIMARY:
            ds=GraphTaskView(view,t)
            if len(ds)<2:raise ValueError('too few training observations')
            batches[t]=DataCollator()([ds[0],ds[1]])
            ids[t]=[ds.get_sample_id(0),ds.get_sample_id(1)]
        rows=one_step(args,source,batches,scaler.trainer_scalers(),output,device=a.device)
        result=dict(scope='ROUTE_B_REAL_WEIGHT_ONE_STEP_SMOKE_NOT_FORMAL_TRAINING',source=source_receipt,
            input_identity=table.identity,train_sample_ids=ids,scaler=scaler.to_dict(),runs=rows,
            saved_tensor_verification=verify_step_records(output,rows),
            device=a.device,test_predictions_accessed=False,calibration_predictions_accessed=False,
            selection_performed=False,formal_training_ready=False)
        with (output/'smoke_result.json').open('x',encoding='utf8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        print('ROUTE_B_ONE_STEP_SMOKE_COMPLETE: B0/B1/RPT; PENDING_CODEX_REVIEW')
    except Exception as exc:
        parser.exit(2,f'ROUTE_B_SMOKE_FAILED: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':main()
