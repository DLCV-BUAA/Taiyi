import math

from .base_class import SingleStepQuantity
from ...extensions import ForwardInputEigOfCovExtension
from ..utils.calculation import *
import torch
import numpy as np
import pdb

class InputCovCondition80(SingleStepQuantity):
    def _compute(self, global_step):
        eig_values, step = getattr(self._module, 'eig_values', (None, None))
        if eig_values is None or step is None or step != global_step:
            data = self._module.input_eig_data
            cov = cal_cov_matrix(data)
            eig_values = cal_eig(cov)
            eig_values, _ = torch.sort(eig_values, descending=True)
            setattr(self._module, 'eig_values', (eig_values, global_step))
        # pdb.set_trace()
        length = len(eig_values)
        index = math.floor(length * 0.8)
        eps =  1e-7
        condition80 = eig_values[0] / (torch.abs(eig_values[index]) + eps)
        # print(eig_values)
        return condition80

    def forward_extensions(self):
        extensions = [ForwardInputEigOfCovExtension()]
        return extensions


if __name__ == '__main__':
    import torch
    from torch import nn as nn

    l = nn.Linear(2, 3)
    cov = nn.Conv2d(2, 2, 3, 1, 1)
    x = torch.randn((4, 2))
    x_c = torch.randn((4, 2, 3, 3))
    quantity_l = InputCovCondition80(l)
    quantity_c = InputCovCondition80(cov)

    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)
    for hook in quantity_c.forward_extensions():
        cov.register_forward_hook(hook)

    for j in range(1):
        y = l(x)
        y_c = cov(x_c)
        quantity_l.track(j)
        quantity_c.track(j)
    print(quantity_l.get_output()[0])
    print(quantity_c.get_output()[0])