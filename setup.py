#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import sys

# if sys.version_info < (3, 6):
#     sys.exit('Sorry, Python >= 3.6 is required for deepund.')

setup(
    name='Taiyi',
    version='0.1.0',
    description='A Toolkit for understanding the training of neural network',
    classifiers=[
        # 发展时期,常见的如下
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # 开发的目标用户
        'Intended Audience :: Science/Research',
        # 许可证信息
        'License :: OSI Approved :: MIT License',
        # 目标 Python 版本
        'Programming Language :: Python :: 3.8',
        # 属于什么类型
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    # setup.py 本身要依赖的包，这通常是为一些setuptools的插件准备的配置
    # 这里列出的包，不会自动安装。
    setup_requires=[
    ],
    # 表明当前模块依赖哪些包，若环境中没有，则会从pypi中下载安装
    install_requires=[
        'wandb',
        'plotly',
    ],
    packages=find_packages(),
)
