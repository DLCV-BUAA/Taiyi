from .base_class import SingleStepQuantity
from ...extensions import ForwardInputEigOfCovExtension
import torch
import pdb

class InputCovMaxEig(SingleStepQuantity):
    def _compute(self, global_step):
        max_eigen_value = torch.max(self._module.input_eig.real)
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