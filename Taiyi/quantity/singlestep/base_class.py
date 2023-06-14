from ..quantity import Quantity


class SingleStepQuantity(Quantity):

    def _should_compute(self, global_step):
        return self._track_schedule(global_step)

    def _compute(self, global_step):
        raise NotImplementedError
