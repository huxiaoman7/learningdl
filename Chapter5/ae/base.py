#-*- coding:utf-8 -*-
'''
Created on 2016��5��16��

@author: Administrator
'''
import numpy as np
#import pandas as pd

#创建神经网络类
# nodes为1*n矩阵，表示每一层有多少个节点
class nn():
    def __init__(self,nodes):
        self.layers = len(nodes)
        self.nodes = nodes;
        # 学习率
        self.u = 1.0;
        # 权值
        self.W = list();
        # 偏差值
        self.B = list()
        # 层值
        self.values = list();
        # 误差
        self.error = 0;
        # 损失
        self.loss = 0;

        for i in range(self.layers-1):
            # 权值初始化，权重范围-0.5~0.5
            self.W.append(np.random.random((self.nodes[i],self.nodes[i+1])) - 0.5)   
            # B值初始化
            self.B.append(0)
            # P值初始化
#            self.P.append(0)
        
        for j in range(self.layers):
            # values值初始化
            self.values.append(0)
            
            
#创建autoencoder类，可以看成是多个神经网络简单的堆叠而来
class autoencoder():
    def __init__(self):
        self.encoders = list()
    def add_one(self,nn):
        self.encoders.append(nn)
    