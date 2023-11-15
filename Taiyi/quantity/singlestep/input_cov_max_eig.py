from .base_class import SingleStepQuantity
from ...extensions import ForwardInputEigOfCovExtension
from ..utils.calculation import *
import torch
import pdb

class InputCovMaxEig(SingleStepQuantity):
    def _compute(self, global_step):
        eig_values, step = getattr(self._module, 'eig_values', (None, None))
        if eig_values is None or step is None or step != global_step:
            data = self._module.input_eig_data
            cov = cal_cov_matrix(data)
            eig_values = cal_eig(cov)
            eig_values, _ = torch.sort(eig_values, descending=True)
            setattr(self._module, 'eig_values', (eig_values, global_step))
        max_eigen_value = eig_values[0]
        return max_eigen_value

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
    quantity_l = InputCovMaxEig(l)
    quantity_c = InputCovMaxEig(cov)

    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)
    for hook in quantity_c.forward_extensions():
        cov.register_forward_hook(hook)

    for i in range(1):
        y = l(x)
        y_c = cov(x_c)
        quantity_l.track(i)
        quantity_c.track(i)
    print(quantity_l.get_output()[0])
    print(quantity_c.get_output()[0])