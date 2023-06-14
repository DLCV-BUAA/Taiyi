from .base_class import MultiStepQuantity
from ...extensions import ForwardInputExtension
import torch

# from Taiyi.quantity.multistep.base_class import MultiStepQuantity
# from Taiyi.extensions.forward_extension import ForwardInputExtension

class MeanTID(MultiStepQuantity):

    def _compute_ones(self, global_step):
        data = self._module.input
        if data.dim() == 3:
            data = data.transpose(0, 2).contiguous().view(data.shape[2], -1)
        else:
            data = data.transpose(0, 1).contiguous().view(data.shape[1], -1)
        return data.mean(dim=1)
    
    def _compute(self, global_step):
        # self.catch 是列表， 列表的每个元素是每个batch的均值
        diff_data = [d - self._module.running_mean for d in self.cache]
        diff_data = torch.stack(diff_data, dim=0)
        # diff_data = torch.tensor([d - self._module.running_mean for d in self.cache])
        eps = 1e-8
        result = diff_data.norm(dim=-1) / (torch.sqrt(self._module.running_var).data.norm(dim=-1) + eps)
        return result.mean()

    def forward_extensions(self):
        extensions = [ForwardInputExtension()]
        return extensions


if __name__ == '__main__':
    import torch
    from torch import nn as nn
    
    

    l = nn.BatchNorm1d(3)
    # cov = nn.Conv2d(1, 2, 3, 1, 1)
    x = torch.randn((4, 3, 3))
    # print(x.transpose(0, 1).contiguous().view(x.shape[1], -1).mean(dim=1))
    # x_c = torch.randn((4, 3, 3, 3))
    from Taiyi.utils.schedules import linear
    quantity_l = MeanTID(l, linear(2, 0))
    # quantity_c = InputMean(cov)
    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)
    # for hook in quantity_c.forward_extensions():
    #     cov.register_forward_hook(hook)

    for i in range(3):
        y = l(x)
        # y_c = cov(x_c)
        quantity_l.track(i)
        # quantity_c.track(i)

    print(quantity_l.get_output()[2])
    print(quantity_l.should_show(2))
