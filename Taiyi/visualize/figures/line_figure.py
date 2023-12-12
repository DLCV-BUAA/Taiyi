from .base_figure import *

class LineFigure(Figure):
    def _plot(self):
        for x, y, label in zip(self.x_data, self.y_data, self.legend):
            self.ax.plot(x, y, label=label)
            
   
# xs = []
# ys = []
# legend = []        
# x = np.linspace(0, 2 * np.pi, 100)

# # 绘制30条线，使用新的颜色和样式循环
# for i in range(3):
#     xs.append(x+0.1*i)
#     y = np.sin(x + i * np.pi / 15)
#     ys.append(y)
#     legend.append(str(i))
# data = {}
# data['x'] = xs
# data['y'] = ys
# data['x_label'] = 'sin'
# data['y_label'] = 'cos'
# data['title'] = 'sin_cos'
# data['legend'] = legend

# figure = LineFigure(data)
# figure.plot()
# figure.show()