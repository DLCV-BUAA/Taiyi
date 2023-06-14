from collections import defaultdict

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ..utils.schedules import ScheduleSelector, linear
from ..utils.regisiter import Regisiter
from ..quantity import QuantitySelector


class Monitor:
    def __init__(self, model, configer=None):
        if isinstance(model, DistributedDataParallel):
            self.model = model.module
        else:
            self.model = model
        self.params = list(model.parameters())
        self.output = defaultdict(dict)
        self.configer = configer
        self.parse_module, self.parse_quantity = self._config_parser()
        if len(self.configer) != 0:
            self._register()

    def track(self, global_step):
        for _, quantities in self.parse_quantity.items():
            for quantity in quantities:
                quantity.track(global_step)
                

    def get_output(self):
        self._update_output()
        return self.output

    def clean_mem(self):
        self.get_output().clear()
        for _, quantities in self.parse_quantity.items():
            for quantity in quantities:
                quantity.clean_mem()
                
    def _parse_quantities(self, quantities):
        quantities_list = []
        track_schedule_list = []
        for item in quantities:
            if isinstance(item, str):
                quantities_list.append(item)
                track_schedule_list.append(linear())
            else:
                if len(item) == 1:
                    track_schedule_list.append(linear())
                else:
                    track_schedule_list.append(ScheduleSelector.select(item[1]))
                quantities_list.append(item[0])
                
        return zip(quantities_list, track_schedule_list)
    

    def _config_parser(self):
        parse_module = defaultdict(dict)
        parse_quantity = defaultdict(dict)
        for module_name, quantities in self.configer.items():
            try:
                module = self._get_submodule(module_name)
                module.name = module_name
                parse_module[module_name] = module
                parse_quantity[module_name] = [QuantitySelector.select(quantity)(module, track_schedule) for quantity,  track_schedule in
                                               self._parse_quantities(quantities)]
            except (AttributeError, TypeError):
                for name, module in self.model.named_modules():
                    if self._is_module(module_name, module) and name not in parse_module.keys():
                        module.name = name
                        parse_module[name] = module
                        parse_quantity[name] = [QuantitySelector.select(quantity)(module, track_schedule) for quantity,  track_schedule in
                                               self._parse_quantities(quantities)]
        return parse_module, parse_quantity

    def _get_submodule(self, target):
        if target == "":
            return self.model

        atoms = target.split(".")
        mod = self.model

        for item in atoms:

            if not hasattr(mod, item):
                raise AttributeError(mod.__class__.__name__ + " has no attribute `" + item + "`")
            if isinstance(mod, nn.Sequential) or isinstance(mod, nn.ModuleList):
                mod = mod[int(item)]
            else:
                mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not an nn.Module")

        return mod

    def _is_module(self, module_name, module):
        if isinstance(module_name, str):
            if module.__class__.__name__ == module_name:
                return True
            else:
                return False
        elif isinstance(module_name, type):
            if isinstance(module, module_name):
                return True
            else:
                return False
        else:
            return False

    def _register(self):
        for module_name, quantities in self.parse_quantity.items():
            module = self.parse_module[module_name]
            forward_extensions = self._process_duplicate_extensions(
                [quantity.forward_extensions() for quantity in quantities])
            if len(forward_extensions) > 0:
                Regisiter.register_forward(module, forward_extensions)

            backward_extensions = self._process_duplicate_extensions(
                [quantity.backward_extensions() for quantity in quantities])
            if len(backward_extensions) > 0:
                Regisiter.register_backward(module, backward_extensions)

    def _update_output(self):
        for module_name, quantities in self.parse_quantity.items():
            for quantity in quantities:
                self.output[module_name][quantity.__class__.__name__] = quantity.get_output()

    def _process_duplicate_extensions(self, extensions):
        ext_dict = dict()
        no_duplicate_ext = []
        for es in extensions:
            for extension in es:
                if type(extension) in ext_dict:
                    pass
                else:
                    no_duplicate_ext.append(extension)
                    ext_dict[type(extension)] = True

        return no_duplicate_ext


