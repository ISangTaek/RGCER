import csv
from pathlib import Path
import numpy as np
import pytest

from pgt_v2 import preparation
from pgt_v2.artifacts import load_arrays, load_manifest, read_json
from pgt_v2.cache import file_sha256
from pgt_v2.config import CONFIG_ROOT


def synthetic_source(tmp_path,monkeypatch,*,invalid=False):
    tasks=['rat_oral_LD50','mouse_oral_LD50']
    source=tmp_path/'synthetic.csv'
    fields=['TAID','SMILES','smiles','InChIKey','Pubchem CID',*tasks]
    with source.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i in range(20):
            smiles='C'*(i+1)
            if invalid and i==0:smiles='invalid'
            writer.writerow(dict(TAID=f'SYN-{i}',SMILES=smiles,smiles=smiles,InChIKey='',**{'Pubchem CID':'',tasks[0]:i*.2,tasks[1]:i*.1}))
    rules=read_json(CONFIG_ROOT/'preparation_rules.json')
    rules.update(source_sha256=file_sha256(source),candidate_task_names=tasks)
    rules['target_block']['qualified_human_priority']=[]
    rules['endpoint_differences']['ordered_candidates']=[tasks]
    original=preparation.read_json
    def read(path):
        name=Path(path).name
        if name=='preparation_rules.json':return rules
        if name in ('expected_parse_failures.json','input_overrides.json'):return {}
        return original(path)
    monkeypatch.setattr(preparation,'read_json',read)
    # Structural clustering itself has separate C02 real-chemistry tests. This
    # fixture isolates the materializer with 20 independent synthetic groups.
    monkeypatch.setattr(preparation,'structure_groups',lambda molecules:{r:'g_'+m.model_input_id for r,m in molecules.items()})
    return source


def test_materialize_five_fold_views_and_nested_masks(tmp_path,monkeypatch):
    source=synthetic_source(tmp_path,monkeypatch)
    output=tmp_path/'prepared'
    result=preparation.prepare(source,output)
    assert result['real_fits']==0 and result['selected_outer_folds']==5 and result['views']==35
    full=output/'training_views/F0/full'
    small=output/'training_views/F0/P010_U1042';large=output/'training_views/F0/P030_U1042'
    a=load_arrays(small,'train.npz',load_manifest(small))['labels']
    b=load_arrays(large,'train.npz',load_manifest(large))['labels']
    assert np.all(~np.isfinite(a)|np.isfinite(b))
    assert (small/'validation.npz').read_bytes()==(full/'validation.npz').read_bytes()
    assert (small/'test_inputs.npz').read_bytes()==(full/'test_inputs.npz').read_bytes()
    assert read_json(output/'checksums.json')['cache/graphs.sqlite3']==file_sha256(output/'cache/graphs.sqlite3')


def test_new_parse_failure_stops_before_cache_or_views(tmp_path,monkeypatch):
    source=synthetic_source(tmp_path,monkeypatch,invalid=True)
    output=tmp_path/'failed'
    with pytest.raises(ValueError,match='New chemistry failures'):preparation.prepare(source,output)
    assert read_json(output/'execution.json')['status']=='STOP'
    assert len(read_json(output/'failures.json'))==1
    assert not (output/'cache').exists() and not (output/'training_views').exists()
