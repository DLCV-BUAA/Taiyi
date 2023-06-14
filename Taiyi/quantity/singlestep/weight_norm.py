from .base_class import SingleStepQuantity


class WeightNorm(SingleStepQuantity):

    def _compute(self, global_step):
        data = self._module.weight
        return data.norm(2)
