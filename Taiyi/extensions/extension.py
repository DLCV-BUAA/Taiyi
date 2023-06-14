from abc import ABC
import torch

class Extension(ABC):
    """
    extension的基类，作用是将hook得到的结果值作为module的属性进行保存，不同的module在获取相同值的时候对应的hook操作不同
    子类在继承基类时，要实现_default hook：当子类没有实现未知的nn.module时，返回的默认的hook

    在quantity计算时，如果某个中间结果复用性足够强，可以考虑将中间结果作为extension实现
    """
    def _get_module_extension_hook(self, module):
        try:
            hook = getattr(self,  '_' + module.__class__.__name__)
        except AttributeError:
            hook = self._default
        return hook

    @torch.no_grad()
    def __call__(self, module, input, output):
        module_hook = self._get_module_extension_hook(module)
        if module_hook is not None:
            result = module_hook(module, input, output)
            setattr(module, self._name, result)

    def _default(self, module, input, output):
        raise NotImplementedError
