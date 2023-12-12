"""
结果可视化类，暂时通过第三方软件进行展示
"""
from collections import defaultdict
from ..figures import Surface3d

import os
import json
import numpy as np


class Visualization:
    def __init__(self, monitor, visualize,dir='./output', project='task', name='name'):
        self.clean_step = 500
        self.monitor = monitor
        self.vis = visualize
        self.dir = dir
        self.project = project
        self.name = name
        self.save_dir = os.path.join(self.dir, self.project, self.name)
        if not os.path.exists(self.save_dir):
            # 判断当前output文件夹是否存在
            os.makedirs(self.save_dir)

    def show(self, step, ext=None):
        """
        1. 获取module_name
        2. 获取module_name 对应的 quantity
        2. 获取结果值 module:quantity:epoch
        :return:
        """
        logs = defaultdict(dict)
        save_logs = defaultdict(dict)
        module_names = self._get_module_name()
        for module_name in module_names:
            quantitis = self.monitor.parse_quantity[module_name]
            quantity_names = self._get_quantity_name(module_name)
            for quantity, quantity_name in zip(quantitis, quantity_names):
                if not quantity.should_show(step):
                    continue
                key = module_name + '_' + quantity_name
                val = self._get_result(module_name, quantity_name, step)
                save_logs[key] = val
                if val.size == 1:
                    val = val.item()
                else:
                    val = self._get_result(module_name, quantity_name)
                    val = Surface3d(val, key)
                logs[key] = val
        if ext is not None:
            logs.update(ext)
        self.vis.log(logs)
        self.save_to_local(step, save_logs)
        # if step % self.clean_step == 0:
        #     self.monitor.clean_mem()

    
    def save_to_local(self, step=0, data_log=None, log_type='monitor'):
        if data_log is not None and len(data_log) != 0:
            self._save(step, data_log, log_type)
    
    def _save(self, step, data_log, log_type):
        # print(data_log)
        data_log['step'] = step
        for key in data_log:
            if isinstance(data_log[key], np.ndarray):
                data_log[key] = data_log[key].tolist() 
        file_name = os.path.join(self.save_dir, log_type + '_' + str(step) + '.json')
        with open(file_name, 'w') as f:
            json.dump(data_log, f)
    
    def log_ext(self, step=None, ext=None, log_type='train'):
        self.vis.log(ext)
        self.save_to_local(step, ext, log_type)

    def close(self):
        self.vis.finish()
        return

    def _get_module_name(self):
        module_names = self.monitor.get_output().keys()
        return module_names

    def _get_quantity_name(self, module_name):
        quantity_name = self.monitor.get_output()[module_name].keys()
        return quantity_name

    def _get_result(self, module_name, quantity_name, step=None):
        if step != None:
            value = self.monitor.get_output()[module_name][quantity_name][step]
        else:
            value = self.monitor.get_output()[module_name][quantity_name]
        return value
