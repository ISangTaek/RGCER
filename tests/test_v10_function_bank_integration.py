"""Real Graphormer code with synthetic datasets; no external assets required."""
import pytest
import torch

from reproducibility import state_dict_sha256
from tests.test_v9_s3_screen import joint
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime, one_cpu_thread, setup_factory
import v9_s2_screen as s2
from v10_functional_transfer import FrozenFunctionBank


@pytest.mark.parametrize('setting', ('ToxAcute','A','B'))
def test_graph_function_bank_does_not_use_batch_labels_or_mutate_source(setting,joint):
    factory,trainer,source,_=joint(setting)
    encoder_hash=state_dict_sha256(trainer.model.encoder)
    head_hash=state_dict_sha256(source.heads)
    bank=FrozenFunctionBank(trainer.model.encoder,source.heads)
    task=next(iter(s2.datasets_for(factory,trainer,setting)['train']))
    batch=s2.batch_for(factory,trainer,setting,'train',task,[0],'cpu')
    h,p=bank(batch)
    original_y=batch.y.clone()
    batch.y.fill_(-123456.)
    altered_h,altered_p=bank(batch)
    batch.y.copy_(original_y)
    torch.testing.assert_close(h,altered_h,atol=0,rtol=0)
    torch.testing.assert_close(p,altered_p,atol=0,rtol=0)
    assert h.shape[0]==p.shape[0]==1 and p.shape[1]==len(source.tasks)
    assert bank.tasks==tuple(source.tasks) and not p.requires_grad
    assert state_dict_sha256(trainer.model.encoder)==encoder_hash
    assert state_dict_sha256(source.heads)==head_hash
