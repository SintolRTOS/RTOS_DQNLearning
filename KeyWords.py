# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:43:07 2019

@author: wangjingyi
"""

import sys
import numpy as np
import tensorflow as tf
sys.path.append('./')
sys.path.append('model')

from WordAgent import WordAgent
from Util import Memory ,StateProcessor
from DDPG import DDPG
from ACNetwork import ACNetwork
np.random.seed(1)
tf.set_random_seed(1)
import time
import logging  # 引入logging模块
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
logger.setLevel(level = logging.DEBUG)
handler = logging.FileHandler("KeyWords/log/log_" + str(time.time()) + '.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)
# 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)

class DDPG4KeyWords(DDPG):
    """docstring for ClassName"""
    def __init__(self, **kwargs):
        super(DDPG4KeyWords, self).__init__(**kwargs)
        
    def _build_a_net(self,s,scope,trainable):

        #w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        w_initializer, b_initializer = None,None
        with tf.variable_scope(scope):
            e1 = tf.layers.dense(inputs=s, 
                    units=30, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation = tf.nn.relu,
                    trainable=trainable)  
            a = tf.layers.dense(inputs=e1, 
                    units=self.n_actions, 
                    bias_initializer = b_initializer,
                    kernel_initializer=w_initializer,
                    activation = tf.nn.tanh,
                    trainable=trainable) 

        return tf.multiply(a, self.a_bound, name='scaled_a')
    
    def _build_c_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.n_features, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.n_actions, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a) 

############################ keword ddpg model parameters #####################
    
MAX_EPISODES = 1000000
MAX_EP_STEPS = 2
batch_size = 32

evn = WordAgent('assert/keyword.xlsx','xlsx')
evn.openExcel()
evn.reset()

n_features = evn.get_observation().shape[0]
n_actions = 1
a_bound = np.array([2.])
memory_size = 10000

logger.info('n_features: ' + str(n_features))
logger.info('n_actions: ' + str(n_actions))
logger.info('a_bound: ' + str(a_bound))

############################### training ######################################

ddpg = DDPG4KeyWords(n_actions=n_actions,
        n_features=n_features,
        reward_decay=0.9,
        lr_a = 0.001,
        lr_c = 0.002,
        TAU = 0.01,
        output_graph=False,
        log_dir = 'KeyWords/log/DDPG4KeyWords/',
        a_bound =a_bound,
        model_dir = 'KeyWords/model_dir/DDPG4KeyWords/')
memory = Memory(memory_size = memory_size)
var = 3
t1 = time.time()
step = 0
for i in range(MAX_EPISODES):
    s = evn.reset()
    logger.info('env.reset() s: ')
    logger.info(str(s))
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        step+=1
        
        #add explorate noise
        logger.info('ddpg.choose_action s: ')
        logger.info(str(s))
        a = ddpg.choose_action(s)
        logger.info('action choose: ' + str(a))
        a = np.clip(np.random.normal(a, var), -2, 2)
        logger.info('var action choose: ' + str(a))
        s_,r,done,info = evn.step(a)
        memory.store_transition(s,a,r,s_)
        
        if step > memory_size:
            logger.info('-----------DDPG4KeyWords learning :-----------------' + str(step))
            var *= .9995
            logger.info('learn var: ' + str(var))
            data = memory.sample(batch_size)
            logger.info('learn data: ' + str(data))
            ddpg.learn(data)
            
        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            logger.info('result: ' + str(evn.result))
            result_keywords = evn.get_result_keywords()
            logger.info('Episode:' + str(i) + ' Reward: ' + str(ep_reward) + ' Result: ' + str(result_keywords))
            break
logger.info('Running time: ' +  str(time.time() - t1))    
        
        
        