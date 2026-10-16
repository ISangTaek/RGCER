"""Explicitly synthetic labels; no toxicity source or historical cache reads."""
from pathlib import Path
import numpy as np

from architecture.toxacute_tasks import TOXACUTE_TASKS
from pgt_v2.artifacts import save_view, write_json
from pgt_v2.cache import CacheEntry, build_cache
from pgt_v2.chemistry import prepare_molecule, digest, framed_digest
from pgt_v2.data import make_training_view


def synthetic_bundle(root, tasks=TOXACUTE_TASKS):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    smiles=['C','CC','CCC','CCCC','CO','CCO','CN','CCN','CF','CCF','CCl','CCCl']
    labels=np.array([[i*.3+j*.02+np.sin(i+j)*.5 for j in range(len(tasks))] for i in range(12)])
    for i in range(12): labels[i,(i*3)%len(tasks)]=np.nan
    source=digest(dict(kind='SYNTHETIC_NOT_TOXICITY',smiles=smiles,label_rule='i*.3+j*.02+sin(i+j)*.5; selected missing'))
    molecules=[prepare_molecule(s) for s in smiles]
    entries=[CacheEntry(m,f'synthetic_{i}',i,{'source':'synthetic'}) for i,m in enumerate(molecules)]
    cache=build_cache(entries,root/'cache',source_sha256=source)
    records=[dict(record_id=e.sample_id,raw_row_index=i,cache_offset=i,chemical_id=e.molecule.model_input_id,
                  group_id='g_'+e.molecule.model_input_id) for i,e in enumerate(entries)]
    roles=['train']*6+['validation']*3+['test']*3
    mask=np.isfinite(labels)&(np.asarray(roles)=='train')[:,None]
    view=make_training_view(labels=labels,label_task_names=tasks,task_names=tasks,
        record_ids=[r['record_id'] for r in records],raw_row_indices=list(range(12)),
        group_ids=[r['group_id'] for r in records],chemical_ids=[r['chemical_id'] for r in records],
        roles=roles,visible_mask=mask,source_sha256=source,fold_id='F0')
    manifest=save_view(root/'view',view,cache_manifest=cache,records=records,roles=roles,labels=labels,fold='F0')
    by_chemical={r['chemical_id']:r['record_id'] for r in records[:6]}
    ordered=sorted(by_chemical,key=lambda c:(framed_digest(['F0',c]),c))
    write_json(root/'reference.json',dict(fold='F0',source_sha256=source,record_ids=[by_chemical[c] for c in ordered],chemical_ids=ordered))
    return manifest
