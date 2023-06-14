<h1 align="center">debug tools</h1>
<p align="right">作者：贾俊龙
<p align="right">时间：2023/3/6


> demoV1.0
>
> 

[TOC]

## 1. 运行流程

```python
# 伪代码
def prepare_config():
    config = {'epoch': 100, 'w': 10, 'h': 10, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_taiyi = {
        'conv1': ['InputSndNorm', 'OutputGradSndNorm', 'WeightStatistic'],
        'l1': ['InputSndNorm', 'OutputGradSndNorm', 'WeightStatistic']
    }
    return config, config_taiyi

if __name__ == '__main__':
    config, config_taiyi = prepare_config()
    prepare_data()
    model = Model()
    prepare_optimizer()
    prepare_loss_func()
    
    *******************************************
    taiyi = Taiyi(model, config_taiyi)        *
    *******************************************
    
    for epoch in range(config['epoch']):
        opt.zero_grad()
        y_hat = model(x)
        loss = loss_fun(y_hat, y)
        loss.backward()
        *************************************************
        taiyi.track(epoch)                              *
        log_tools.log(taiyi)                            *
        *************************************************
        opt.step()
    *******************************************
    taiyi.get_output()                        *
    *******************************************
```



## 2. 模块

### 2.1 Taiyi

#### 2.1.1 主要功能

​	获取需要监控的module以及需要观察的quantity

​	对module进行quantity需要的hook进行注册

​	进行quantity计算

​	获取quantity计算得到的结果

#### 2.1.2 接口

​	`__init__(configer)`:

			- _parse_configer
			- _regisiter	

​	`track`：计算指标值

​	`get_output`:获得输出

#### 2.1.3 与其他模块的交流

​	`utils`:**schedule**，**regisiter**

​	`quantity` :**QuantitySelector**

### 2.2 Quantity

#### 2.2.1 主要功能

进行指标计算，一个quantity实例仅对一个module有用	

#### 2.2.2 接口

​	`__init__(module)`

​	`forward_extensions`:向外界提供forward_extensions

​	`backward_extensions`:向外界提供backward_extensions

​	`track`:计算指标

​	`get_output`:向外界提供计算结果

### 2.3 Extension

#### 2.2.1 主要功能

​	hook

#### 2.2.2 接口

​	`__call__`:

			1. get_hook(module)
   			2. result = hook(module, input, output)
   			3. module.attribute = result