import math

from .base_class import SingleStepQuantity
from ...extensions import ForwardInputEigOfCovExtension
import torch
import pdb

class InputCovCondition80(SingleStepQuantity):
    def _compute(self, global_step):
        # pdb.set_trace()
        eig_values = self._module.input_eig.real.numpy()
        length = len(eig_values)
        eig_values.sort()
        index = -math.ceil(length * 0.8)
        condition80 = eig_values[index:]
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