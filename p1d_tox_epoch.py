"""P1D Tox epoch bridge; no CLI, asset lookup, selection or training authorization.

Keep the original Trainer sampler, eager loader iterators and epoch RNG policy.
The caller supplies a bound factory trainer and its original train loaders.
"""
import math
import torch
from p1d_optimization import require
from p1d_tox import TASKS


def run_epoch(trainer, loaders, epoch):
    require(type(epoch) is int and 0 <= epoch < 40, 'epoch scope')
    require(list(trainer.task_name)==list(TASKS) and list(loaders)==list(TASKS),'Human3 task order')
    require(trainer.args.task_sampling=='proportional' and trainer.args.tasks_per_update==1,
            'original proportional one-task update required')
    require(trainer.args.weighting=='EW' and not list(trainer.loss_balancer.parameters()),'plain EW required')
    require(all(loader.batch_size==64 and not loader.drop_last for loader in loaders.values()),'original batch policy')
    counts={t:len(loader) for t,loader in loaders.items()}
    samples={t:len(loader.dataset) for t,loader in loaders.items()}
    require(all(counts[t]==math.ceil(samples[t]/64) and samples[t]>0 for t in TASKS),'loader population')
    expected=sum(counts.values());observed=0
    before=int(trainer.optimizer_updates)
    control=trainer.optimization
    if control is not None:
        require(trainer.optimizer is control.optimizer,'controlled optimizer binding')
        control.begin_epoch(epoch)

    def before_step(*_):
        require(observed<expected,'epoch update cap')
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.model.parameters()),
                'nonfinite gradient before update')
        if control is not None:control.verify_frozen()

    def after_step(*_):
        nonlocal observed
        observed+=1

    pre=trainer.optimizer.register_step_pre_hook(before_step)
    post=trainer.optimizer.register_step_post_hook(after_step)
    try:
        result=trainer._train_epoch(loaders,epoch)
    finally:
        pre.remove();post.remove()
    require(observed==expected==result['updates']==trainer.optimizer_updates-before,'actual epoch updates')
    require(result['schedule_usage']==counts,'actual task batch counts')
    require(result['schedule_diagnostics']['actual_task_samples']==samples,'actual train sample exposure')
    require(all(math.isfinite(result['loss'][t]) for t in TASKS),'nonfinite task loss')
    require(all(torch.isfinite(v).all() for v in trainer.model.state_dict().values()),'nonfinite trained model')
    if control is not None:control.verify_frozen()
    return dict(epoch=epoch,updates=observed,cumulative_updates=trainer.optimizer_updates,
        task_batches=counts,task_samples=samples,loss=result['loss'],
        optimization=None if control is None else control.record())
