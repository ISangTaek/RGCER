"""Regression for 071: compact graph indices are not original CSV row numbers."""
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from toxacute_datastore import ToxAcuteDataStore, ToxAcuteTaskDataset, build_datastore_v2
from v9_srgt import TrainSupport, training_records


def test_real_datastore_with_invalid_middle_row_joins_by_stable_id(tmp_path):
    raw=tmp_path/'raw.csv'
    pd.DataFrame(dict(TAID=['a','bad','c','d','e','f'],
        smiles=['CC','not_a_smiles','CCC','CCO','CCN','CCCl'],
        task_a=[1.,2.,3.,4.,5.,6.])).to_csv(raw,index=False)
    root=tmp_path/'data'
    build_datastore_v2(raw,root,task_names=['task_a'],splitting='random',
        valid_size=0.,calibration_size=0.,test_size=0.,lmdb_map_size_gb=.001,max_path_distance=6)
    store=ToxAcuteDataStore.resolve(root)
    try:
        manifest=json.loads((store.root/'split_manifest.json').read_bytes())
        by_id={r['sample_id']:r for r in manifest['records']}
        assert list(store.row_indices)==[0,1,2,3,4]
        assert sorted(r['row_index'] for r in manifest['records'])==[0,2,3,4,5]
        ds=ToxAcuteTaskDataset(store,task_name='task_a',split='train')
        # Old code chooses raw CSV row 2 for global graph 2 (actually row 3),
        # reproducing 071's sample-ID mismatch before a single update.
        ds.indices=np.asarray([2,4])
        old_row=next(r for r in manifest['records'] if r['row_index']==2)
        assert old_row['sample_id']!=ds.get_sample_id(0)
        f=SimpleNamespace(store=store,datasets={'train':{'task_a':ds}})
        records=training_records(f,None,'ToxAcute')
        assert len(records)==2
        support=TrainSupport(records)
        for i,record in enumerate(records):
            graph=ds[i]
            trusted=by_id[ds.get_sample_id(i)]
            assert record['sample_id']==graph.sample_id==trusted['sample_id']
            assert record['canonical']==graph.canonical_smiles==trusted['canonical_smiles']
            assert record['group']==trusted['split_group']
            assert record['split']=='train'
            assert support.for_train_batch('task_a',[graph.sample_id],[graph.canonical_smiles]).shape==(1,2)
    finally:store.close()


def metadata_fixture(tmp_path):
    # Deliberately reversed manifest order and noncontiguous original row IDs.
    rows=[dict(row_index=7,sample_id='s7',split='train',canonical_smiles='CCC',split_group='g7',num_nodes=3),
          dict(row_index=0,sample_id='s0',split='train',canonical_smiles='CC',split_group='g0',num_nodes=2)]
    store=SimpleNamespace(root=tmp_path,row_indices=[0,1],sample_ids=['s0','s7'],split_codes=[0,0],num_nodes=[2,3])
    ds=SimpleNamespace(indices=[1],split='train',get_sample_id=lambda i:'s7')
    factory=SimpleNamespace(store=store,datasets={'train':{'t':ds}})
    return rows,factory,ds


def test_manifest_order_and_raw_row_numbers_do_not_control_the_join(tmp_path):
    rows,f,_=metadata_fixture(tmp_path)
    (tmp_path/'split_manifest.json').write_text(json.dumps(dict(records=rows)),encoding='utf8')
    assert training_records(f,None,'ToxAcute')==[dict(task='t',sample_id='s7',canonical='CCC',group='g7',split='train')]


@pytest.mark.parametrize('bad',['duplicate_manifest_id','missing_id','extra_id','duplicate_store_id',
                               'wrong_dataset_id','wrong_split_code','holdout_record','wrong_dense_index','negative_index'])
def test_identity_corruption_remains_fail_closed(tmp_path,bad):
    rows,f,ds=metadata_fixture(tmp_path)
    if bad=='duplicate_manifest_id':rows.append(dict(rows[0]))
    if bad=='missing_id':rows.pop(0)
    if bad=='extra_id':rows.append(dict(rows[0],sample_id='extra'))
    if bad=='duplicate_store_id':f.store.sample_ids[0]='s7'
    if bad=='wrong_dataset_id':ds.get_sample_id=lambda i:'s0'
    if bad=='wrong_split_code':f.store.split_codes[1]=3
    if bad=='holdout_record':rows[0]['split']='test';f.store.split_codes[1]=3
    if bad=='wrong_dense_index':f.store.row_indices[1]=7
    if bad=='negative_index':ds.indices=[-1]
    (tmp_path/'split_manifest.json').write_text(json.dumps(dict(records=rows)),encoding='utf8')
    with pytest.raises(ValueError):training_records(f,None,'ToxAcute')
