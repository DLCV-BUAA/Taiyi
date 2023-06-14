from ..extension import Extension


class ForwardInputExtension(Extension):
    """
    获取当前module的输入值，并将结果保存到module.input字段中
    """
    def __init__(self):
        self._name = 'input'
        super(ForwardInputExtension, self).__init__()

    def _default(self, module, input, output):
        return input[0]

    def _Linear(self, module, input, output):
        return input[0]

    def _Conv2d(self, module, input, output):
        return input[0]

if __name__ == '__main__':
    # from torch.nn import Linear
    # import torch
    # l = Linear(10, 10)
    # forward_input_extension = ForwardInputExtension()
    # x = torch.randn((5, 10))
    # l.register_forward_hook(forward_input_extension)
    # y = l(x)
    # print(x)
    # print(l.input)

    from torch.nn import Conv2d
    import torch

    l = Conv2d(in_channels=1, out_channels=3, stride=1, padding=2, kernel_size=3)
    forward_input_extension = ForwardInputExtension()
    x = torch.randn((2, 1, 3, 3))
    l.register_forward_hook(forward_input_extension)
    y = l(x)
    print(x[0])
    print(l.input[0])
