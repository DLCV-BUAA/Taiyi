'''
@Description: 
@Author: jiajunlong
@Date: 2023-12-12 14:32:19
@LastEditTime: 2023-12-12 15:42:43
@LastEditors: jiajunlong
'''
"""
结果可视化类，暂时通过第三方软件进行展示
"""

import os
import json
import numpy as np
import pandas as pd
from ..figures import *
from ..io_utils import *


class LocalVisualization:
    def __init__(self,dir='./output', project='task', save_dir='./output/pictrue'):
        self.dir = dir
        self.task = project
        self.save_dir = save_dir
        self.data_loader = LoadTaskData(root=self.dir, task=self.task)

    def show(self, quantity_name, project_name=None, data_type='monitor'):
        data = self.data_loader.load_data(quantity_name, project_name, data_type)
        figure = self._select_figure(data)
        figure.plot()
        figure.show()
        
    def _select_figure(self, data):
        return LineFigure(data)

    
    def save(self, quantity_name, project_name=None, data_type='monitor'):
        data = self.data_loader.load_data(quantity_name, project_name, data_type)
        figure = self._select_figure(data)
        figure.plot()
        figure.save()
    
