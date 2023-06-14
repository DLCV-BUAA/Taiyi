from collections import defaultdict

import torch
import torch.nn as nn

from model import  Model
from Taiyi.taiyi.monitor import Monitor
import wandb
from Taiyi.visualize import Visualization


def prepare_data(w, h, class_num, len):
    x = torch.randn((len, 3, w, h), requires_grad=True)
    y = torch.randint(0, class_num, (len,))
    return x, y


def prepare_optimizer(model, lr=1e-2):
    return torch.optim.SGD(model.parameters(), lr=lr)


def prepare_config():
    config = {'epoch': 100, 'w': 224, 'h': 224, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_taiyi = {
        # 'Conv2d': ['InputSndNorm', 'OutputGradSndNorm', 'WeightStatistic'],
        nn.BatchNorm2d: [['MeanTID', 'linear(5,0)'],'InputSndNorm']
    }
    return config, config_taiyi


def prepare_loss_func():
    return nn.CrossEntropyLoss()


if __name__ == '__main__':
    config, config_taiyi = prepare_config()
    print(config_taiyi)
    # wandb.init(project='dubug-tools', name='test_quantity_remove_hooksmodule3', config=config)
    x, y = prepare_data(config['w'], config['h'], config['class_num'], config['len'])
    model = Model(config['w'], config['h'], config['class_num'])
    # model = Model()
    opt = prepare_optimizer(model, config['lr'])
    loss_fun = prepare_loss_func()
    #######################################
    monitor = Monitor(model, config_taiyi)
    # vis = Visualization(monitor, wandb)
    #######################################
    for epoch in range(config['epoch']):
        opt.zero_grad()
        y_hat = model(x)
        loss = loss_fun(y_hat, y)
        loss.backward()
        ##############################
        monitor.track(epoch)
        # vis.show(epoch)
        print(epoch)
        ##############################
        opt.step()
    # vis.close()
    # print(monitor.get_output())