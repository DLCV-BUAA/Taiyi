from ..extension import Extension


class BackwardOutputExtension(Extension):
    """
    获取当前module的输出的梯度，并将结果保存到module.output_grad字段中
    """
    def __init__(self):
        self._name = 'output_grad'

    def _default(self, module, grad_input, grad_output):
        return grad_output[0]

    def _Linear(self, module, grad_input, grad_output):
        return grad_output[0]

    def _Conv2d(self, module, grad_input, grad_output):
        return grad_output[0]


if __name__ == '__main__':
    # from torch.nn import Linear
    # import torch
    #
    # l = Linear(10, 10)
    # backward_output_extension = BackwardOutputExtension()
    # x = torch.randn((2, 10), requires_grad=True)
    # y = torch.randint(0, 10, (2,))
    # l.register_full_backward_hook(backward_output_extension)
    # y_hat = l(x)
    # y_hat.retain_grad()
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss = loss_fn(y_hat, y)
    # loss.backward()
    # print(y_hat.grad)
    # print(l.output_grad)

    import torch.nn as nn
    import torch

    l = nn.Conv2d(in_channels=1, out_channels=3, stride=1, padding=1, kernel_size=3)
    backward_output_extension = BackwardOutputExtension()
    x = torch.randn((2, 1, 4, 3), requires_grad=True)
    y = torch.randint(0, 3, (2,))
    l.register_full_backward_hook(backward_output_extension)
    y_hat = l(x)
    y_hat.retain_grad()
    yy_hat = y_hat.view(2, -1)
    yy_hat = yy_hat.sum(1)
    # print(y_hat)
    loss = sum(y - yy_hat)
    loss.backward()
    print(y_hat.grad)
    print(l.output_grad)
