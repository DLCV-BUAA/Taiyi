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
    def __init__(self, data, figsize=(16, 10), dpi=80, default_save_dir='./output/pictrue'):
        self.x_data = data['x']
        self.y_data = data['y']
        self.x_label = data['x_label']
        self.y_label = data['y_label']
        self.legend = data['legend']
        self.title = data['title']
        self.figsize = figsize
        self.dpi = dpi
        self.ax = None
        self.default_save_dir = default_save_dir
        
    def plot(self):
        self._pre_plot()
        self._plot()
        self._finalize_plot() 
    
    def show(self):
        plt.show()
    
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
        plt.savefig(file_path)


    def _plot(self):
        raise NotImplementedError("Subclasses must implement the '_plot' method.")
    
    def _pre_plot(self):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = plt.gca()
    
    def _finalize_plot(self):
        self._set_xy()
        self._set_title()
        self._remove_borders()
        self.ax.legend()
    
    def _set_title(self):
        self.ax.set_title(self.title)
    
    def _set_xy(self): 
        # 设置 X 轴和 Y 轴标签
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)

    
    def _remove_borders(self):

        # Remove borders
        self.ax.spines["top"].set_alpha(0.0)
        self.ax.spines["bottom"].set_alpha(0.3)
        self.ax.spines["right"].set_alpha(0.0)
        self.ax.spines["left"].set_alpha(0.3)


    
    
    



    