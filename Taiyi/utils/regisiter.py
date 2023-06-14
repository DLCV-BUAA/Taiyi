from collections import defaultdict


class Regisiter:
    """
    对模块进行hook注册
    可能需要将hook remove掉
    """
    def __init__(self):
        self.forward_handles = defaultdict()

    @staticmethod
    def register_forward(model, forward_hooks):
        for hook in forward_hooks:
            model.register_forward_hook(hook)

    @staticmethod
    def register_backward(model, backward_hooks):
        for hook in backward_hooks:
            model.register_full_backward_hook(hook)


