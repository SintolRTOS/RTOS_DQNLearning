# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:35:05 2019

@author: wangjingyi
"""

import sys
import gym
import numpy as np 
import tensorflow as tf
sys.path.append('./')
sys.path.append('model')
 
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
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("Pendulum/log/log_" + str(time.time()) + '.txt')
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



class DDPG4Pendulum(DDPG):
    """docstring for ClassName"""
    def __init__(self, **kwargs):
        super(DDPG4Pendulum, self).__init__(**kwargs)
    

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
   
    
    # def _build_c_net(self,s,a,scope,trainable):
    #     #trainable = True if reuse is None else False
    #     w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

    #     #with tf.variable_scope('Critic',reuse = reuse,custom_getter=custom_getter):
    #     with tf.variable_scope(scope):
    #         s1 = tf.layers.dense(inputs=s, 
    #                 units=32, 
    #                 bias_initializer = b_initializer,
    #                 kernel_initializer=w_initializer,
    #                 activation = tf.nn.relu,
    #                 trainable=trainable)  
    #         a1 = tf.layers.dense(inputs=a, 
    #                 units=32, 
    #                 bias_initializer = b_initializer,
    #                 kernel_initializer=w_initializer,
    #                 activation = tf.nn.relu,
    #                 trainable=trainable) 

    #         h_dense = s1+a1#tf.concat([s1, a1], axis=1, name="h_concat")
          
         
    #         # h_dense  = tf.layers.dense(inputs=h_dense, 
    #         #         units=16, 
    #         #         bias_initializer = b_initializer,
    #         #         kernel_initializer=w_initializer,
    #         #         activation = tf.nn.relu,
    #         #         trainable=trainable)
    #         q  = tf.layers.dense(inputs=h_dense, 
    #                 units=1, 
    #                 bias_initializer = b_initializer,
    #                 kernel_initializer=w_initializer,
    #                 activation = tf.nn.relu,
    #                 trainable=trainable)

    #     return q   

#####################  hyper parameters  ####################

MAX_EPISODES = 50000
MAX_EP_STEPS = 500
batch_size  = 32
RENDER = True
ENV_NAME = 'Pendulum-v0'

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

print('env.observation_space: ' + str(env.observation_space))
print('env.action_space: ' + str(env.action_space))


n_features = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
a_bound = env.action_space.high
memory_size = 10000

logger.info('n_features: ' + str(n_features))
logger.info('n_actions: ' + str(n_actions))
logger.info('a_bound: ' + str(a_bound))
###############################  training  ####################################


ddpg = DDPG4Pendulum(n_actions=n_actions,
        n_features=n_features,
        reward_decay=0.9,
        lr_a = 0.001,
        lr_c = 0.002,
        TAU = 0.01,
        output_graph=False,
        log_dir = 'Pendulum/log/DDPG4Pendulum/',
        a_bound =a_bound,
        model_dir = 'Pendulum/model_dir/DDPG4Pendulum/')
memory = Memory(memory_size=memory_size)
var = 3  # control exploration
t1 = time.time()
step = 0
for i in range(MAX_EPISODES):
    s = env.reset()
    logger.info('env.reset() s: ' + str(s))
    ep_reward = 0
    logger.debug('env s:')
    logger.debug(s)
    for j in range(MAX_EP_STEPS):
        step+=1
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        logger.info('action choose: ' + str(a))
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        logger.info('var action choose: ' + str(a))
        s_, r, done, info = env.step(a)

        memory.store_transition(s, a, r / 10, s_)
        logger.info('memory.store_transition: ' + str(s) + ' ,' + str(a) + ' ,' + str(r/10) + ' ,' + str(s_))

        if step > memory_size:
                #env.render()
                logger.info('-----------DDPG4Pendulum learning :-----------------' + str(step))
                var *= .9995    # decay the action randomness
                logger.info('learn var: ' + str(var))
                data = memory.sample(batch_size)
#                logger.info('learn data: ' + str(data))
                ddpg.learn(data)

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            logger.info('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
logger.info('Running time: ', time.time() - t1)
