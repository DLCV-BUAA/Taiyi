# import torch.nn as nn
# import torch
#
# l = nn.Linear(10, 10)
#
#
# class hooktest:
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#
#     def __call__(self, model, input, output):
#         print(input)
#         print(self.a)
#         print(self.b)
#
#     def _Linear(self, a):
#         print(a)
#
#
# # l.register_forward_hook(hooktest(1, 2))
# # x = torch.randn((100, 10))
# # y = l(x)
#
# hook = hooktest(1, 2)
# fun = getattr(hook, "_" + l.__class__.__name__)
# print(l.__class__.__name__)
# fun(5)


# class a:
#     def __init__(self):
#         ...
#
#     def __call__(self, *args, **kwargs):
#         print('aaa')
#
# b = a()
# b.name = 'bn'
# b
import torch
l = torch.tensor(
    [
        [2.0, 3, 4, 5],
        [1, 2, 3, 4],
    ]
)
print(l.var(dim=1))

