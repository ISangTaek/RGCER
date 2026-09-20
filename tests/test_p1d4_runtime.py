import pytest
import p1d4_runtime as runtime
from p1d_schedule import job
from tests.test_p1d_tox import fixture_factory
from tests.test_p1d_routes import setup_factory


@pytest.mark.parametrize('setting',['ToxAcute','A','B'])
def test_real_synthetic_forty_epoch_runtime_and_readonly_reverification(setting,tmp_path,monkeypatch,setup_factory):
    if setting=='ToxAcute':factory=fixture_factory()
    else:factory=setup_factory[0](setting)[0]
    monkeypatch.setattr(runtime,'factory_for',lambda *a,**kw:factory)
    j=job(setting,'B1_low',42,'screen')
    options=dict(split_manifest=tmp_path/'split',source_lock=tmp_path/'source',device='cpu')
    result=runtime.execute_job(tmp_path,j,tmp_path/'run',**options)
    assert result==runtime.verify_job(tmp_path,j,tmp_path/'run',**options)
    assert len(result['result']['history'])==40
    assert all('r2' in ep for r in result['result']['history'] for ep in r['endpoints'].values())
