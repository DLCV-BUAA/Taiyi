from .base_class import MultiStepQuantity
import torch


class WeightParamJump(MultiStepQuantity):

    def _compute_ones(self, global_step):
        data = self._module.weight
        return data
    
    def _compute(self, global_step):
        if len(self.cache) == 1:
            return 0
        jump_num = [d * d_p < 0 for d, d_p in zip(self.cache[-1], self.cache[-2])]
        return sum(jump_num)
