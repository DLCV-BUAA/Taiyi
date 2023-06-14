from .singlestep import *
from .multistep import *


class QuantitySelector:
    """
    静态类， 可以通过str获取quantity
    好处：
    """
    @staticmethod
    def select(quantity_name):
        if quantity_name not in globals():
            raise NotImplementedError(
                "hook not found: {}".format(quantity_name))
        quantity = globals()[quantity_name]
        return quantity
