#-*- coding:utf-8 -*-
import autoencoder_train
from base import autoencoder,nn
import numpy as np

x = np.array([[0,0,1,0,0],
            [0,1,1,0,1],
            [1,0,0,0,1],
            [1,1,1,0,0],
            [0,1,0,1,0],
            [0,1,1,1,1],
            [0,1,0,0,1],
            [0,1,1,0,1],
            [1,1,1,1,0],
            [0,0,0,1,0]])
y = np.array([[0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [1],
            [0]])
#################################
# step1 建立autoencoder
#弄两层autoencoder
nodes=[5,3,2]
#建立auto框架
ae = autoencoder_train.aebuilder(nodes)
#设置部分参数
#训练
ae = autoencoder_train.aetrain(ae, x, 6000)
##############################
# step2 微调
#建立完全体的autoencoder
nodescomplete = np.array([5,3,2,1])
aecomplete = nn(nodescomplete)
for i in range(len(nodescomplete)-2):
    aecomplete.W[i] = ae.encoders[i].W[0]
    
aecomplete = autoencoder_train.nntrain(aecomplete, x, y, 6000)
print aecomplete.values[3]
