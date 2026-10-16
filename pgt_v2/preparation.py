"""G01 materialization only: registered structures, presence-based splits and views.

No model is fitted here. Unexpected failures stop publication and are retained.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from molecular_features import PGT_FEATURE_SPEC
from .artifacts import read_json, save_view, write_json
from .cache import CacheEntry, build_cache, file_sha256
from .chemistry import framed_digest, resolve_input_record
from .config import CONFIG_ROOT
from .data import make_training_view, sparse_visible_mask
from .splits import structure_groups, plan_folds


def write_tsv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def read_source(source, tasks):
    with Path(source).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames) != len(set(reader.fieldnames)) or set(tasks) - set(reader.fieldnames):
            raise ValueError('Duplicate CSV columns or missing registered tasks')
        rows = list(reader)
    labels = np.full((len(rows),len(tasks)), np.nan)
    for i,row in enumerate(rows):
        for j,task in enumerate(tasks):
            token = row[task].strip()
            if token:
                value = float(token)
                if not np.isfinite(value):
                    raise ValueError(f'Nonempty invalid label row={i} task={task}')
                labels[i,j] = value
    return rows, labels


def inventory(source, output):
    rules = read_json(CONFIG_ROOT/'preparation_rules.json')
    checksum = file_sha256(source)
    if checksum != rules['source_sha256']:
        raise ValueError('Source is not the registered C01 data')
    tasks = rules['candidate_task_names']
    rows, labels = read_source(source,tasks)
    report = dict(source_sha256=checksum, records=len(rows), task_names=tasks,
        finite_labels=int(np.isfinite(labels).sum()), per_task=np.isfinite(labels).sum(0).tolist(),
        feature_spec=PGT_FEATURE_SPEC.metadata(), fitted_models=0)
    write_json(output,report)
    return report


def prepare(source, output):
    rules = read_json(CONFIG_ROOT/'preparation_rules.json')
    sha = file_sha256(source)
    if sha != rules['source_sha256']:
        raise ValueError('Source SHA mismatch; cannot replace with another author dataset')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    tasks=rules['candidate_task_names']
    rows, labels=read_source(source,tasks)
    overrides=read_json(CONFIG_ROOT/'input_overrides.json')
    expected_failures=read_json(CONFIG_ROOT/'expected_parse_failures.json')
    molecules, entries, records, failures = {}, [], [], []
    status=dict(status='RUNNING',source_sha256=sha,real_fits=0,scientific_acceptance='CODEX_REVIEW_PENDING')
    write_json(output/'execution.json',status)
    write_json(output/'source_manifest.json',dict(sha256=sha,records=len(rows),labels=int(np.isfinite(labels).sum()),task_names=tasks))
    write_json(output/'feature_spec.json',PGT_FEATURE_SPEC.metadata())
    try:
        for i,row in enumerate(rows):
            name=f'row_{i}'
            raw=dict(record_id=name,TAID=row['TAID'],raw_smiles_upper=row['SMILES'],selected_input_smiles=row['smiles'],
                source_inchikey=row['InChIKey'],pubchem_cid=row['Pubchem CID'])
            try:
                molecule,provenance=resolve_input_record(raw,source_sha256=sha,overrides=overrides)
            except Exception as error:
                known=expected_failures.get(name)
                allowed=known is not None and known['TAID']==row['TAID'] and known['smiles']==row['smiles'] and Chem.MolFromSmiles(row['smiles']) is None
                failures.append(dict(record_id=name,raw_row_index=i,TAID=row['TAID'],reason=str(error),
                    error_type=type(error).__name__,finite_labels=int(np.isfinite(labels[i]).sum()),registered_parse_exclusion=allowed))
                continue
            molecules[name]=molecule
            records.append(dict(record_id=name,raw_row_index=i,cache_offset=len(entries),chemical_id=molecule.model_input_id,
                nonstereo_id=molecule.nonstereo_id,TAID=row['TAID'],num_atoms=molecule.mol.GetNumAtoms(),
                raw_to_model=json.dumps(molecule.raw_to_model),model_to_raw=json.dumps(molecule.model_to_raw),
                symmetry_classes=json.dumps(molecule.symmetry_classes)))
            entries.append(CacheEntry(molecule,name,i,provenance))
            if (i+1)%1000==0:
                print(json.dumps(dict(stage='chemistry',rows_seen=i+1,successes=len(entries),failures=len(failures))),flush=True)
        write_json(output/'failures.json',failures)
        write_tsv(output/'record_identity.tsv',records,list(records[0]) if records else ['record_id'])
        if any(not x['registered_parse_exclusion'] for x in failures) or {x['record_id'] for x in failures} != set(expected_failures):
            raise ValueError('New chemistry failures or registered exclusion mismatch; no silent dropping')
        cache_manifest=build_cache(entries,output/'cache',source_sha256=sha)
        # Release the second references to large Mol objects as soon as possible.
        entries.clear()
        groups=structure_groups(molecules)
        for record in records:
            record['group_id']=groups[record['record_id']]
        write_tsv(output/'record_identity.tsv',records,list(records[0]))
        labels=labels[[r['raw_row_index'] for r in records]]
        observed=np.isfinite(labels)
        grouped=[r['group_id'] for r in records]
        split=plan_folds(grouped,observed,seed=rules['split_seed'])
        (output/'split_candidates').mkdir()
        for attempt in split['attempts']:
            k=attempt['outer_folds']
            write_json(output/f'split_candidates/folds_{k}.json',attempt)
        write_json(output/'split_summary.json',split)
        coverage=[]
        for attempt in split['attempts']:
            for fold in attempt['folds']:
                if fold['roles'] is None:
                    continue
                for role in ('train','validation','test'):
                    role_mask=np.asarray(fold['roles'])==role
                    for j,task in enumerate(tasks):
                        mask=role_mask & observed[:,j]
                        coverage.append(dict(outer_folds=attempt['outer_folds'],fold=fold['fold_id'],role=role,task=task,
                            rows=int(mask.sum()),chemicals=len({r['chemical_id'] for r,keep in zip(records,mask) if keep}),
                            groups=len({g for g,keep in zip(grouped,mask) if keep})))
        write_tsv(output/'coverage.tsv',coverage,['outer_folds','fold','role','task','rows','chemicals','groups'])
        if split['status'] != 'COVERAGE_PASS_CANDIDATE':
            raise ValueError('Registered 5/3-fold algorithm did not meet all task coverage; C04 review required')
        selected=split['attempts'][-1]
        minimums={t:min(x['groups'] for x in coverage if x['outer_folds']==selected['outer_folds'] and x['role']=='train' and x['task']==t) for t in tasks}
        human=rules['target_block']['qualified_human_priority']
        target=human+sorted(set(tasks)-set(human),key=lambda t:(minimums[t],t))[:3]
        write_json(output/'target_block_candidate.json',dict(tasks=target,ranking_full_train_minimum_groups=minimums,
                   status='CANDIDATE_C04_REVIEW_REQUIRED'))
        pair_reports=[]
        for left,right in rules['endpoint_differences']['ordered_candidates']:
            paired=observed[:,tasks.index(left)] & observed[:,tasks.index(right)]
            counts=[]
            for fold in selected['folds']:
                train=paired & (np.asarray(fold['roles'])=='train'); test=paired & (np.asarray(fold['roles'])=='test')
                counts.append(dict(fold=fold['fold_id'],train_paired_chemicals=len({r['chemical_id'] for r,k in zip(records,train) if k}),
                    test_paired_groups=len({g for g,k in zip(grouped,test) if k})))
            oof=len({g for g,k in zip(grouped,paired) if k})
            pair_reports.append(dict(left=left,right=right,oof_paired_groups=oof,folds=counts,
                eligible=oof>=20 and all(x['train_paired_chemicals']>=10 and x['test_paired_groups']>=1 for x in counts)))
        write_json(output/'endpoint_pairs_candidate.json',pair_reports)
        view_reports=[];fold_rows=[]
        for fold in selected['folds']:
            fold_id='F'+str(fold['fold_id']); roles=fold['roles']
            for record,role in zip(records,roles):
                fold_rows.append(dict(record_id=record['record_id'],raw_row_index=record['raw_row_index'],group_id=record['group_id'],fold=fold_id,role=role))
            settings=[('full',1.,None)]+[(f'P{round(f*100):03d}_U{s}',f,s) for f in (.1,.3) for s in (1042,1043,1044)]
            for name,fraction,subset_seed in settings:
                mask=sparse_visible_mask(observed=observed,task_names=tasks,roles=roles,chemical_ids=[r['chemical_id'] for r in records],
                    target_tasks=target,fraction=fraction,fold_id=fold_id,subset_seed=subset_seed or 1042)
                view=make_training_view(labels=labels,label_task_names=tasks,task_names=tasks,
                    record_ids=[r['record_id'] for r in records],raw_row_indices=[r['raw_row_index'] for r in records],
                    group_ids=grouped,chemical_ids=[r['chemical_id'] for r in records],roles=roles,
                    visible_mask=mask,source_sha256=sha,fold_id=fold_id)
                path=output/'training_views'/fold_id/name
                manifest=save_view(path,view,cache_manifest=cache_manifest,records=records,roles=roles,labels=labels,fold=fold_id)
                view_reports.append(dict(fold=fold_id,name=name,view_id=view.view_id,manifest_sha256=manifest['manifest_sha256'],
                    fraction=fraction,subset_seed=subset_seed,counts=view.count.tolist(),groups=view.group_count.tolist(),
                    active_rows=manifest['active_train_rows'],validation_label_cost=int(observed[np.asarray(roles)=='validation'].sum())))
            # Same full-training reference records for every method and sparse view.
            unique={}
            for record,role in zip(records,roles):
                if role=='train':
                    unique.setdefault(record['chemical_id'],record['record_id'])
            ordered=sorted(unique,key=lambda c:(framed_digest([fold_id,c]),c))[:512]
            write_json(output/'training_views'/fold_id/'reference.json',dict(fold=fold_id,record_ids=[unique[c] for c in ordered],
                chemical_ids=ordered,source_sha256=sha,selection='label-free hash-ranked full-training chemical identities'))
        write_tsv(output/'fold_roles.tsv',fold_rows,['record_id','raw_row_index','group_id','fold','role'])
        write_json(output/'training_views.json',view_reports)
        status.update(status='PREPARED_C04_REVIEW_PENDING',valid_records=len(records),valid_labels=int(observed.sum()),
            groups=len(set(grouped)),cache_id=cache_manifest['cache_id'],selected_outer_folds=selected['outer_folds'],views=len(view_reports))
    except Exception as error:
        status.update(status='STOP',error_type=type(error).__name__,error=str(error))
        write_json(output/'execution.json',status)
        raise
    write_json(output/'execution.json',status)
    # Compact files and full cache content index are returned. Large cache remains server-side.
    checks={str(p.relative_to(output)).replace('\\','/'):file_sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'checksums.json',checks)
    return status
