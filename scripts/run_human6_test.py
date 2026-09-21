"""068-accepted Human6 best checkpoints: validation replay then fixed test export.

No optimizer, training, source loading, HPO, calibration or model selection.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import human6 as h
import numpy as np
import torch

ACCEPTED_SHA = '77b07fc8a2071f481b3c0b5ce37073d9f031f98aab342e6a3f9e1435907d939a'
TRAINING_COMMIT = 'cdbf5a09e817ef82753fe58b8d5966906f56058f'


def accepted_runs(path, p):
    h.require(h.digest(path) == ACCEPTED_SHA, 'not the accepted 068 canonical ZIP')
    with zipfile.ZipFile(path) as z:
        h.require(json.loads(z.read('runtime_policy.json')) == p, 'training policy changed')
        runs = []
        for m in h.METHODS:
            for seed in range(42,47):
                base = f'formal/{m}_s{seed}/'
                r = json.loads(z.read(base+'result.json'))
                r['accepted_validation'] = json.loads(z.read(base+'best_validation.json'))
                h.require(r['identity']['method']==m and r['identity']['seed']==seed, 'run identity')
                runs.append(r)
    return runs


def load_best(path, r):
    f = r['files']['best.pt']
    h.require(path.is_file() and path.stat().st_size==f['size_bytes'] and h.digest(path)==f['sha256'], 'best file SHA/size')
    b = torch.load(path,map_location='cpu',weights_only=True)
    h.require(b['identity']==r['identity'] and b['epoch']==r['best_epoch'], 'best identity/epoch')
    state = b['model_state']
    h.require(h.state_digest(state)==b['model_state_sha'] and all(torch.isfinite(v).all() for v in state.values()), 'best tensor digest/finite')
    return state


def predict(model, view, scaler, device):
    from dataset import DataCollator
    model.eval(); rows=[]
    with torch.inference_mode():
        for j,t in enumerate(h.TASKS):
            ds=h.GraphTaskView(view,t)
            for start in range(0,len(ds),32):
                ids=list(range(start,min(start+32,len(ds))))
                batch=DataCollator()([ds[i] for i in ids])
                h.require(not batch.is_empty and batch.y.numel()==len(ids), 'graph row dropped')
                raw=model(batch.to(device),task_name=t)[t]
                h.require(raw.shape==(len(ids),3) and torch.isfinite(raw).all(), 'nonfinite/shape prediction')
                pred=(raw[:,0].double()*scaler.stds[j]+scaler.means[j]).cpu().tolist()
                for i,v in zip(ids,pred):
                    k=ds.indices[i]
                    rows.append(dict(task=t,split=view.split,sample_id=view.sample_ids[k],
                        canonical=view.canonical[k],group=view.groups[k],label=float(view.labels[k,j]),prediction=v))
    return rows


def compare_replay(actual, expected):
    key=lambda r:(r['task'],r['sample_id'])
    a={key(r):r for r in actual}; e={key(r):r for r in expected}
    h.require(len(a)==len(actual)==len(expected)==len(e) and a.keys()==e.keys(), 'replay population')
    for k,r in a.items():
        q=e[k]
        h.require({k:v for k,v in r.items() if k!='prediction'}=={k:v for k,v in q.items() if k!='prediction'}, 'replay truth/metadata')
        h.require(math.isfinite(r['prediction']) and math.isclose(r['prediction'],q['prediction'],abs_tol=1e-5,rel_tol=1e-6), 'validation replay differs')


def load_test(csv_path, split_path, p):
    """Called only after ALL 15 validation replays passed. Parse only test Human6."""
    h.require(h.digest(csv_path)==p['data']['csv_sha256'] and h.digest(split_path)==p['data']['split_sha256'], 'test input SHA')
    manifest=json.loads(Path(split_path).read_text(encoding='utf-8'))
    h.require(manifest['source_csv_sha256']==p['data']['csv_sha256'], 'manifest CSV')
    refs=h.manifest_records(manifest); records=[]; values=[]; n=0
    with Path(csv_path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); h.task_columns(reader.fieldnames)
        for i,row in enumerate(reader):
            h.require(i in refs and row['smiles']==refs[i]['raw_smiles'] and None not in row and all(v is not None for v in row.values()), 'test CSV mapping')
            n+=1
            if refs[i]['split']!='test': continue
            y=[np.nan if (v:=h.label(row[t])) is None else v for t in h.TASKS]
            if not np.isfinite(y).any(): continue
            records.append(refs[i]); values.append(y)
    h.require(n==len(refs) and h.digest(csv_path)==p['data']['csv_sha256'] and h.digest(split_path)==p['data']['split_sha256'], 'test input changed')
    y=np.asarray(values,dtype=np.float64)
    h.require(y.shape==(48,6) and dict(zip(h.TASKS,np.isfinite(y).sum(axis=0).tolist()))==p['counts']['test'], 'test counts')
    h.require(len({r['split_group'] for r in records})==35, 'test group count')
    y.setflags(write=False)
    return SimpleNamespace(split='test',tasks=h.TASKS,sample_ids=tuple('dataset115:'+r['sample_id'] for r in records),
        smiles=tuple(r['raw_smiles'] for r in records),canonical=tuple(r['canonical_smiles'] for r in records),
        groups=tuple(r['split_group'] for r in records),labels=y)


def test_metrics(rows, view):
    h.require(view.split=='test' and all(r['split']=='test' for r in rows), 'test split')
    # Reuse exact arithmetic/truth join without widening the training View or API.
    met=h.metrics([dict(r,split='validation') for r in rows],view)
    met['known_route_macro_rmse']=sum(met['endpoints'][t]['rmse'] for t in h.TASKS[:-1])/5
    met['known_route_macro_mae']=sum(met['endpoints'][t]['mae'] for t in h.TASKS[:-1])/5
    met['groups']=len(set(view.groups)); met['molecules']=len(view.sample_ids)
    return met


def run(args):
    import subprocess
    from scripts.run_human6 import gpu_free
    h.require(args.gpu in range(4), 'physical GPU0-3')
    gpu_free(args.gpu)
    # This process has not queried CUDA before setting physical mapping.
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    device='cuda:0'; p=h.policy(); runs=accepted_runs(args.accepted_zip,p)
    out=Path(args.output).resolve(); h.require(not out.exists(), 'output exists: no automatic repeat')
    csv_path=Path(args.csv) if args.csv else ROOT/p['data']['csv_path']
    views=h.load_views(csv_path,args.split_manifest,p)
    out.mkdir(parents=True)
    initial=dict(accepted_sha256=ACCEPTED_SHA,training_commit=TRAINING_COMMIT,
        inference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        test_started=False,validation_replayed=0,test_completed=0,physical_gpu=args.gpu)
    h.json_write(out/'progress.json',initial)
    # Verify every exact best before any test numerical access.
    states=[]
    for r in runs:
        d=Path(r['run_directory']) if not args.formal_root else Path(args.formal_root)/Path(r['run_directory']).name
        states.append(load_best(d/'best.pt',r))
    models=[]
    for r,state in zip(runs,states):
        ident=r['identity']; seed=ident['seed']
        h.require(ident['scaler']==h.TaskScaler.fit(views['train'].labels,h.TASKS,allow_empty=False).to_dict(), 'train-only scaler')
        model=h.model_from_source(ident['architecture'],{k[8:]:v for k,v in state.items() if k.startswith('encoder.')},seed)
        model.load_state_dict(state,strict=True); model.to(device)
        scaler=h.TaskScaler.from_dict(ident['scaler'])
        rows=predict(model,views['validation'],scaler,device)
        compare_replay(rows,r['accepted_validation'])
        name=f"{ident['method']}_s{seed}"
        h.json_write(out/(name+'_validation.json'),rows)
        models.append(model.cpu())
        initial['validation_replayed']+=1; h.json_write(out/'progress.json',initial)
    # No selection is performed on validation: all 15 fixed bests advance.
    test=load_test(csv_path,args.split_manifest,p)
    for v in views.values():
        h.require(not set(v.groups)&set(test.groups), 'test/train-validation overlap')
    initial['test_started']=True; h.json_write(out/'progress.json',initial)
    summary=[]
    for r,model in zip(runs,models):
        ident=r['identity']; name=f"{ident['method']}_s{ident['seed']}"
        rows=predict(model.to(device),test,h.TaskScaler.from_dict(ident['scaler']),device)
        met=test_metrics(rows,test); model.cpu()
        h.json_write(out/(name+'_test.json'),rows)
        summary.append(dict(run_id=name,method=ident['method'],seed=ident['seed'],best_epoch=r['best_epoch'],
            checkpoint=r['files']['best.pt'],metrics=met))
        initial['test_completed']+=1; h.json_write(out/'progress.json',initial)
    h.json_write(out/'summary.json',dict(**initial,runs=summary,validation_status='PASS',acceptance_status='PENDING_CODEX_REVIEW',
        csv_sha256=p['data']['csv_sha256'],split_sha256=p['data']['split_sha256'],training_updates=0,calibration_accessed=False))
    print('COMPLETE: 15 validation replays, 15 test exports, 885 test observations; no training')


def package(output, evidence):
    out=Path(output).resolve(); ev=Path(evidence).resolve()
    summary=json.loads((out/'summary.json').read_text())
    h.require(summary['test_completed']==summary['validation_replayed']==15, 'incomplete export')
    paths={n:out/n for n in ('progress.json','summary.json')}
    for m in h.METHODS:
        for seed in range(42,47):
            for split in ('validation','test'):
                n=f'{m}_s{seed}_{split}.json'; paths[n]=out/n
    for n in ('wsl_commands.log','wsl_tests.xml','server_commands.log','server_tests.xml'):
        paths['evidence/'+n]=ev/n
    checks={n:h.digest(path) for n,path in paths.items()}
    dest=out.parent/(out.name+'_review.zip')
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,path in paths.items(): z.write(path,n)
        z.writestr('checksums.json',json.dumps(checks,indent=2))
    with zipfile.ZipFile(dest) as z:
        h.require(z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==s for n,s in checks.items()), 'package integrity')
    with dest.with_suffix('.zip.sha256').open('x') as f: f.write(h.digest(dest)+'  '+dest.name+'\n')
    print(dest)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('run','package'))
    ap.add_argument('--output',required=True)
    ap.add_argument('--accepted-zip'); ap.add_argument('--split-manifest'); ap.add_argument('--csv')
    ap.add_argument('--formal-root'); ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--evidence')
    a=ap.parse_args()
    if a.command=='run':
        h.require(a.accepted_zip and a.split_manifest, 'accepted ZIP and split required'); run(a)
    else:
        h.require(a.evidence,'evidence required'); package(a.output,a.evidence)


if __name__=='__main__': main()
