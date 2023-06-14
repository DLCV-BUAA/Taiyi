"""Base class for a tracked quantity."""

from collections import defaultdict
from ..utils.schedules import linear

import numpy
import torch


class Quantity:
    """
    quantity 是需要观察的指标，quantity类的接口有：
        1. forward_extensions 返回nn.module在前向传播的过程中需要module通过注册hook来获取值的extensions
        2. backward_extensions 返回nn.module在反向传播的过程中需要module通过注册hook来获取值的extensions
        3. track(global_step) 进行指标值的计算
        4. get_output() 输出，形式为{iteration1：result， iteration2：result}
    子类需要实现的接口：
        1. _should_compute(global_step) 返回当前step是否需要计算指标值
        2. _compute(global_step) 对当前step进行指标计算

    需不需要将属性值进行清理
    """

    def __init__(self, module, track_schedule=linear()):

        self._track_schedule = track_schedule
        self._module = module
        self._output = defaultdict(dict)

    def forward_extensions(self):
        return []

    def backward_extensions(self):
        return []

    @torch.no_grad()
    def track(self, global_step):
        if self._should_compute(global_step):
            result = self._compute(global_step)
            if result is not None:
                self._save(global_step, result)

    def get_output(self):
        return self._output

    def clean_mem(self):
        self.get_output().clear()

    def _should_compute(self, global_step):
        """Return if computations need to be performed at a specific iteration.

        Args:
            global_step (int): The current iteration number.

        Raises:
            NotImplementedError: If not implemented. Should be defined by subclass.
        """
        raise NotImplementedError

    def _save(self, global_step, result):
        """Store computation result.

        Args:
            global_step (int): The current iteration number.
            result (arbitrary): The result to be stored.
        """
        self._output[global_step] = self._apply_save_format(result)

    def _apply_save_format(self, value):
        """Apply formatting rules for saved data.

        ``torch.Tensor``s are detached, loaded to CPU and converted to ``numpy`` arrays.
        Items of ``dict``, ``list``, and ``tuple`` are converted recursively.
        ``float``, ``int``, and ``numpy.ndarray`` values are unaffected.

        Args:
            value (Any): Value to be saved.

        Returns:
            Any: Converted value.

        Raises:
            NotImplementedError: If there is no formatting rule for the data type.
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        elif isinstance(value, dict):
            for key, val in value.items():
                value[key] = self._apply_save_format(val)

        elif isinstance(value, list):
            for idx, val in enumerate(value):
                value[idx] = self._apply_save_format(val)

        elif isinstance(value, tuple):
            value = tuple(self._apply_save_format(val) for val in value)

        elif isinstance(value, (float, int, numpy.ndarray)):
            pass

        else:
            raise NotImplementedError(f"No formatting rule for type {type(value)}")

        return value

    def _compute(self, global_step):
        """Evaluate quantity at a step in training.

        Args:
            global_step (int): The current iteration number.

        Raises:
            NotImplementedError: If not implemented. Should be defined by subclass.
        """
        raise NotImplementedError
    
    def should_show(self, global_step):
        return self._track_schedule(global_step)
