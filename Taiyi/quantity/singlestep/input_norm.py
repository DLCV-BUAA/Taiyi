from .base_class import SingleStepQuantity
from ...extensions import ForwardInputExtension


class InputSndNorm(SingleStepQuantity):

    def _compute(self, global_step):
        data = self._module.input
        return data.norm(2)

    def forward_extensions(self):
        extensions = [ForwardInputExtension()]
        return extensions


if __name__ == '__main__':
    import torch
    from torch import nn as nn

    l = nn.Linear(2, 3)
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x = torch.randn((4, 2))
    x_c = torch.randn((4, 1, 3, 3))
    quantity_l = InputSndNorm(l)
    quantity_c = InputSndNorm(cov)
    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)
    for hook in quantity_c.forward_extensions():
        cov.register_forward_hook(hook)

    for i in range(3):
        y = l(x)
        y_c = cov(x_c)
        quantity_l.track(i)
        quantity_c.track(i)
    print(quantity_l.get_output()[0] - x.norm().item())
    print(quantity_c.get_output()[0] - x_c.norm().item())
