from ..extension import Extension


class ForwardOutputExtension(Extension):
    """
    获取当前module的输出值，并将结果保存到module.output字段中
    """
    def __init__(self):
        self._name = 'output'
        super(ForwardOutputExtension, self).__init__()

    def _default(self, module, input, output):
        return output

    def _Linear(self, module, input, output):
        return output

    def _Conv2d(self, module, input, output):
        return output[0]


if __name__ == '__main__':
    from torch.nn import Linear
    import torch
    l = Linear(10, 10)
    forward_output_extension = ForwardOutputExtension()
    x = torch.randn((5, 10))
    l.register_forward_hook(forward_output_extension)
    y = l(x)
    print(y[0])
    print(l.output[0])
    #
    # from torch.nn import Conv2d
    # import torch
    #
    # l = Conv2d(in_channels=1, out_channels=3, stride=1, padding=2, kernel_size=3)
    # forward_output_extension = ForwardOutputExtension()
    # x = torch.randn((2, 1, 3, 3))
    # l.register_forward_hook(forward_output_extension)
    # y = l(x)
    # print(y)
    # print(l.output)
