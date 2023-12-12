from .base_class import SingleStepQuantity


class WeightGradNorm(SingleStepQuantity):

    def _compute(self, global_step):
        data = self._module.weight.grad
        return data.norm(2)