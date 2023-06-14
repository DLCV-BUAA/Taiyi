<h1 align="center">debug tools</h1>
<p align="right">作者：黄雷团队+
<p align="right">时间：2023/3/15

[TOC]

## 1. TODO

- quantity
  - [ ] 输入协方差矩阵的最大特征值
  - [ ] 输入协方差矩阵的稳定秩
  - [ ] 输入协方差矩阵的20%条件数
  - [ ] 输入协方差矩阵的50%条件数
  - [ ] 输入协方差矩阵的80%条件数
- extension
  - forward extension
    - [ ] input协方差矩阵的特征值
  - backward extension
    - [ ] grad_output协方差矩阵的特征值
- taiyi
  - [x] module 解析

## 2.已经实现的功能

- quantity

  |       名称        |       功能       |
  | :---------------: | :--------------: |
  |   InputSndNorm    |   输入的二范数   |
  | OutputGradSndNorm | 输出梯度的二范数 |
  |    WeightNorm     |   权重的二范数   |

  

- extension

  - forward extension

    |             名称              |          功能           |  属性值   |
    | :---------------------------: | :---------------------: | :-------: |
    |     ForwardInputExtension     |        获取输入         |   input   |
    |    ForwardOutputExtension     |        获取输出         |  output   |
    | ForwardInputEigOfCovExtension | input协方差矩阵的特征值 | input_eig |

  - backward extension

    |              名称               |             功能              |     属性值      |
    | :-----------------------------: | :---------------------------: | :-------------: |
    |     BackwardInputExtension      |        获取输入的梯度         |   input_grad    |
    |     BackwardOutputExtension     |        获取输出的梯度         |   output_grad   |
    | BackwardOutputEigOfCovExtension | grad_output协方差矩阵的特征值 | output_grad_eig |

- taiyi
  - 模块解析

## 3. 整体要求 

> - 规范
>   - 每个类的命名规范
>   - 代码编写用有表达力的名字表达含义search、extract这种，循环迭代器用带名字例如xxx_i来代替I
>   - 名字不能有歧义，尤其是针对TRUE 和 FALSE这种，要明确当前函数名称返回的值跟名称的真假性
> - 注释
>   - 注释风格应该一致（最起码在组件内部一致，并在组件中给出注释风格含义）
>   - 每个函数都应该编写注释，注释我觉得可以用中文毕竟是给自己看hhhh，要理解的准确无误
>   - 能直接看出含义（让一个完全不懂项目的人根据代码能知道这段代码表达什么意思）的代码不需要写注释
>     可以用注释记录采用当前解决方法的思考过程以及提醒一些特殊情况：比如TODO（待做），hack（粗糙的解决方案）等等
> - 代码编写
>   - 尽可能使用少的长表达式
>   - 减小变量作用域（全局变量这种，如果是必要的，可以写个静态类专门存储全局变量）
>   - 大问题拆分成小问题，在解决具体方法时先写伪代码，所有伪代码中涉及的细节用专门的小函数进行计算（要注意拆分力度）
>     多用标准库实现

## 4. 团队协作

### 4.1 仓库克隆

- 加入项目以后从自己的工作空间复制克隆仓库的链接

 	1. `git clone XXXXXXXX复制的链接XXXXXXXXXXX`   :   clone仓库到本地
 	2. `git checkout dev`   : 切换到`dev`分支
 	3. `git push -u origin dev`: 首次push到dev分支时使用-u参数指的是设置 up stream，之后push/pull的时候不用再显式指明push到哪个分支

### 4.2 版本管理

#### 4.2.1 名词解释

1. `main`分支，只有确认无误才进行merge
2. `dev`分支，主要的工作分支
3. `jia`分支，本地个人开发分支，名字也可以取为开发新功能的名字

#### 4.2.2 代码管理

1. `git pull`将远程`dev`分支拉取到本地
2. `git switch -c jia`:在本地创建一个新的个人工作分支，在此分支上进行代码编写，bug调试
3. `git add .   /   git commit -m '本次commit内容'  `确认当前功能编写完成并且测试无误，在本地个人分支上可以commit多次
4. `git checkout dev `:切换到开发分支
5. `git merge jia`:将本地个人工作分支 `merge`到本地`dev`分支
6. `git push`:将本地的`dev`分支push到远程仓库

