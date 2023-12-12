from .base_class import SingleStepQuantity
from ...extensions import ForwardInputExtension


class InputStd(SingleStepQuantity):

    def _compute(self, global_step):
        data = self._module.input
        if data.dim() == 3:
            data = data.transpose(0, 2).contiguous().view(data.shape[2], -1)
        else:
            data = data.transpose(0, 1).contiguous().view(data.shape[1], -1)
        # print(type(data.mean(dim=1)))
        return data.std()

    def forward_extensions(self):
        extensions = [ForwardInputExtension()]
        return extensions