# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:41:51 2019

@author: wangjingyi
"""

import os
import numpy as np
import tensorflow as tf
from abc import ABCMeta,abstractmethod
np.random.seed(1)
tf.set_random_seed(1)
print('ACNetwork init tensorflow sucessful!')

import logging  # 引入logging模块
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
# 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = False;
session = tf.Session(config=tfconfig)

class ACNetwork(object):
    __metaclass__ = ABCMeta
    """docstring for ACNetwork"""
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate,
                 memory_size,
                 reward_decay,
                 output_graph,
                 log_dir,
                 model_dir):
        super(ACNetwork,self).__init__()
        
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.output_graph = output_graph
        self.lr = learning_rate
        self.log_dir = log_dir
        self.model_dir = model_dir
        
        #total learning step
        self.learn_step_counter = 0
        
        self.s = tf.placeholder(tf.float32,[None] + self.n_features,name='s')
        self.s_net = tf.placeholder(tf.float32,[None] + self.n_features,name='s_next')
        self.r = tf.placeholder(tf.float32,[None,],name='r')
        self.a = tf.placeholder(tf.int32,[None,],name='a')
        
        with tf.variable_scope('Critic'):
            
        
        
        
        
        
        
        
        
        
        
        
        