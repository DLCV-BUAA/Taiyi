'''
@Description: 
@Author: jiajunlong
@Date: 2023-12-12 14:32:19
@LastEditTime: 2023-12-14 14:23:26
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
    def __init__(self,dir='./output', project='task', save_dir='./output/pictrue', 
                figsize=(16, 10), dpi=80, ncols=3):
        self.dir = dir
        self.task = project
        self.save_dir = save_dir
        self.data_loader = LoadTaskData(root=self.dir, task=self.task)
        self.ncols = 3
        self.figsize = figsize
        self.dpi = dpi
        self.figure = None #plt.figure(figsize=figsize, dpi=dpi)

    def show(self, quantity_name, project_name=None, data_type='monitor'):
        figures = self._plot(quantity_name, project_name,data_type)
        if figures is not None:
            self.figure.show()
        self._clear_figure()
        
    def _select_figure(self, data):
        return LineFigure

    
    def save(self, quantity_name, project_name=None, data_type='monitor',
             save_dir=None, file_name=None, save_type='png', save_subfigures=True):
        figures = self._plot(quantity_name, project_name,data_type)
        if figures is not None:
            dir = save_dir
            if dir is None:
                dir = self.save_dir
            if not os.path.exists(dir):
                os.makedirs(dir)
            file_name = file_name
            if file_name is None:
                file_name = quantity_name + '.' + save_type
            file_path = os.path.join(dir, file_name)
            self.figure.savefig(file_path)
            # print(len(figures))
            if save_subfigures:
                if len(figures) > 1:
                    for figure in figures:
                        # print('save')
                        figure.save()
        self._clear_figure()

        
    def get_project_name(self):
        return self.data_loader.get_project_name()
    
    def get_quantity_name(self, project_name=None, data_type='monitor'):
        return self.data_loader.get_quantity_name(project_name, data_type)
    
    def _cal_gridspe(self, num):
        nrows = num // self.ncols
        add = num % self.ncols != 0
        return nrows + add
    
    def _plot(self, quantity_name, project_name=None, data_type='monitor'):
        figures = []
        selected_quantity_name = []
        for name in self.get_quantity_name(project_name, data_type):
            if quantity_name in name:
                selected_quantity_name.append(name)
        nrows = self._cal_gridspe(len(selected_quantity_name))
        if self.figure is None:
            fcal, frow = self.figsize
            self.figure = plt.figure(figsize=(fcal, frow*nrows), dpi=self.dpi)
        all_handles = []
        all_labels = []
        # print(selected_quantity_name)
        if len(selected_quantity_name) > 1:
            grid = plt.GridSpec(nrows, self.ncols, figure=self.figure)
            for i, name in enumerate(selected_quantity_name):
                data = self.data_loader.load_data(name, project_name, data_type)
                ax = self.figure.add_subplot(grid[i//self.ncols, i%self.ncols])
                figure = self._select_figure(data)(data, ax)
                figure.plot()
                legend = ax.get_legend()
                handles, labels = ax.get_legend_handles_labels()
                for label, handle in zip(labels, handles):
                    if label not in all_labels:
                        all_handles.append(handle)
                        all_labels.append(label)
                legend.remove()  # 移除子图中的legend
                figures.append(figure)
        elif len(selected_quantity_name) == 1:
            data = self.data_loader.load_data(selected_quantity_name[0], project_name, data_type)
            ax = self.figure.add_subplot()
            figure = self._select_figure(data)(data, ax)
            figure.plot()
            figures.append(figure)
        else:
            return None
        self.figure.legend(all_handles, all_labels)
        self.figure.subplots_adjust(wspace=0.5, hspace=0.5)
        return figures
    
    def _clear_figure(self):
        self.figure = None
    
