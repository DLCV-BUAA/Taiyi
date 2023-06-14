from ..quantity import Quantity
from ...utils.schedules import linear
import torch


class MultiStepQuantity(Quantity):
    def __init__(self, module, track_schedule=linear()):
        super(MultiStepQuantity, self).__init__(module, track_schedule)
        self.cache = []
        
        
    def _should_compute(self, global_step):
        return self._track_schedule(global_step)


    def _compute(self, global_step):
        raise NotImplementedError
    
    
    def _compute_ones(self, global_step):
        raise NotImplementedError
    
    @torch.no_grad()
    def track(self, global_step):
        self.cache.append(self._compute_ones(global_step))
        # import sys
        # print(f'cache memory size:{sys.getsizeof(self.cache)/1024/1024}')
        if self._should_compute(global_step):
            result = self._compute(global_step)
            self.cache.clear()
            if result is not None:
                self._save(global_step, result)