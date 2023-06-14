from .base_class import SingleStepQuantity
from ...extensions import BackwardOutputExtension


class OutputGradSndNorm(SingleStepQuantity):

    def _compute(self, global_step):
        data = self._module.output_grad
        return data.norm(2)

    def backward_extensions(self):
        extensions = [BackwardOutputExtension()]
        return extensions


if __name__ == '__main__':
    import torch
    from torch import nn as nn

    # l = nn.Linear(2, 3)
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # x = torch.randn((4, 2), requires_grad=True)
    x_c = torch.randn((4, 1, 3, 3), requires_grad=True)
    # quantity_l = OutputGradSndNorm(l)
    quantity_c = OutputGradSndNorm(cov)
    # for hook in quantity_l.backward_extensions():
    #     l.register_full_backward_hook(hook)
    for hook in quantity_c.backward_extensions():
        cov.register_full_backward_hook(hook)

    for i in range(3):
        # y = l(x)
        # y.retain_grad()
        # sum(sum(y)).backward()
        # quantity_l.track(i)
        y_c = cov(x_c)
        y_c.retain_grad()
        sum(sum(y_c.view(4, -1))).backward()
        quantity_c.track(i)
    # print(quantity_l.get_output()[0] - y.grad.norm().item())
    print(quantity_c.get_output()[0] - y_c.grad.norm().item())
