"""Source-anchored plastic graph branch; a hypothesis, not a superiority claim.

The frozen encoder, trainable pretrained copy and raw-graph branch are combined
as [source + g_delta * (adapted - source), g_graph * target]. The existing
identity/zero fusion and task heads preserve the source function at init.
"""
from copy import deepcopy
import torch
from v9_srgt import DualGraph, require


class PlasticGraph(DualGraph):
    def __init__(self, base, counts):
        super().__init__(base, 'SRGT', counts)
        self.adaptive = deepcopy(base.encoder)
        self.adaptive.requires_grad_(True)
        self.train(True)

    def forward(self, inputs, task_name=None, return_aux=False):
        require(task_name in self.task_name, 'explicit known task required')
        require(not self.encoder.training and all(not p.requires_grad for p in self.encoder.parameters()),
                'anchor must remain frozen/eval')
        with torch.no_grad():
            source = self.encoder(inputs)
        adapted = self.adaptive(inputs)
        target = self.target(inputs)
        gates = self.gates(task_name, getattr(inputs, 'v9_support', None), source)
        representation = self.fusion(torch.cat((source + gates[:, :1]*(adapted-source),
                                                gates[:, 1:]*target), dim=1))
        raw = self.decoders[task_name](representation)
        output = {task_name:raw}
        if not return_aux:return output
        return output, dict(final_raw=raw, base_raw=raw, route_raw=raw,
            final_representation=representation, source_representation=source,
            adapted_representation=adapted, target_representation=target, gates=gates)
