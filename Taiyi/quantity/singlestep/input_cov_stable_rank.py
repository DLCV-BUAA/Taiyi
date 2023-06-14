import math

from .base_class import SingleStepQuantity
from ...extensions import ForwardInputEigOfCovExtension
import torch
import pdb

class InputCovStableRank(SingleStepQuantity):
    def _compute(self, global_step):
        eig_values = self._module.input_eig.real
        max_eigen_value = max(eig_values)
        assert (max_eigen_value != 0), "max_eigen_value can not be zero"
        eigs_sum = float(eig_values.sum())
        stable_rank = eigs_sum / max_eigen_value
        return stable_rank

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
    quantity_l = InputCovStableRank(l)
    quantity_c = InputCovStableRank(cov)

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
    # print(quantity_c.get_output()[0] - x_c.norm().item())