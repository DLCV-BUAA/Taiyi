from .base_class import SingleStepQuantity
from ...extensions import ForwardOutputExtension
import torch


class LinearDeadNeuronNum(SingleStepQuantity):

    def _compute(self, global_step):
        data = self._module.output
        output = data.view(-1, data.shape[-1])
        zero_num = [torch.all(output[:, i] <= -2) for i in range(data.shape[-1])]
        return sum(zero_num)/data.shape[-1]
    
    def forward_extensions(self):
        extensions = [ForwardOutputExtension()]
        return extensions

