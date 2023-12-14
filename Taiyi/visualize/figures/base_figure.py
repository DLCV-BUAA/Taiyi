'''
@Description: 
@Author: jiajunlong
@Date: 2023-12-08 16:37:41
@LastEditTime: 2023-12-14 14:11:32
@LastEditors: jiajunlong
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os



class Figure:
    """
    1. 设置绘图风格  
    2. 选择X坐标以及对应设置 step
    3. 选择Y坐标以及对应设置 quantity
    4. 选择绘图方式[sns/plt]
    5. 确定数据
        {
            'step': [[0, 1 , 2 ...]],
            'quantity': [[1,43,5,56,6,6....]],
            'name': ['name_a']
        }line plot
    """
    def __init__(self, data, ax=None, default_save_dir='./output/pictrue/sub_picture'):
        self.x_data = data['x']
        self.y_data = data['y']
        self.x_label = data['x_label']
        self.y_label = data['y_label']
        self.legend = data['legend']
        self.title = data['title']
        self.figsize=(16, 10)
        self.dpi = 80
        self.ax = ax
        if self.ax is None:
            plt.figure(figsize=(16, 10), dpi=80)
            self.ax = plt.gca()
        self.default_save_dir = default_save_dir
        
    def plot(self, ax=None):
        ax = self._get_ax(ax)
        # self._pre_plot()
        self._plot(ax)
        self._finalize_plot(ax) 
    
    def show(self, ax=None):
        ax = self._get_ax(ax)
        ax.figure.canvas.draw()
    
    def save(self, file_name=None, save_dir=None, save_type='png'):
        dir = save_dir
        if dir is None:
            dir = self.default_save_dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        file_name = file_name
        if file_name is None:
            file_name = self.title + '.' + save_type
        file_path = os.path.join(dir, file_name)
        figure = plt.figure(figsize=(16, 10), dpi=80)
        ax = figure.add_subplot(1, 1, 1)
        self.plot(ax)
        figure.savefig(file_path)
        plt.close()

    def _get_ax(self, ax=None):
        if ax is None:
            return self.ax
        return ax
    
    def _plot(self, ax):
        raise NotImplementedError("Subclasses must implement the '_plot' method.")
    
    def _pre_plot(self):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.gca()
    
    def _finalize_plot(self, ax=None):
        self._set_xy(ax)
        self._set_title(ax)
        self._remove_borders(ax)
        ax.legend()
    
    def _set_title(self, ax=None):
        ax.set_title(self.title)
    
    def _set_xy(self, ax=None): 
        # 设置 X 轴和 Y 轴标签
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

    
    def _remove_borders(self, ax=None):

        # Remove borders
        ax.spines["top"].set_alpha(0.0)
        ax.spines["bottom"].set_alpha(0.3)
        ax.spines["right"].set_alpha(0.0)
        ax.spines["left"].set_alpha(0.3)
        
    def unsetlegend(self, ax=None):
        ax = self._get_ax(ax)
        ax.legend().set_visible(False)
        


    
    
    



    