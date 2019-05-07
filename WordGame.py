# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:35:05 2019

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
handler = logging.FileHandler("WordGame/log/wordgame_log_" + str(time.time()) + '.txt')
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


class KeyWordRank(object):

    def __init__(self):
        super(KeyWordRank,self).__init__()
        self.rank_list = []
        self.rank_value = []
    
    def checkrank(self,rewardvalue,keylist):
#        logger.info('start self.keyword_rank_list: ' + str(self.rank_list))
        rankcout = len(self.rank_value)
        flag = False
        cur_index = 0
        new_rank_value = []
        new_keyword_rank_list = []
        for i in range(rankcout):
            if i >= MAX_SELECT_STEPS:
                break
            value = self.rank_value[i]
            
            if value == rewardvalue and str(keylist) == self.rank_list[i]:
                return
       
            if value < rewardvalue:
                cur_index = i
                flag = True
                break
    
        if flag:
            new_rank_value.clear()
            new_keyword_rank_list.clear()
            for j in range(rankcout):
                if j >= MAX_SELECT_STEPS:
                    break
                if j == cur_index:
                    new_rank_value.append(rewardvalue)
                    new_keyword_rank_list.append(str(keylist))
            
                if len(new_rank_value) < MAX_SELECT_STEPS:
                    new_rank_value.append(self.rank_value[j])
                    new_keyword_rank_list.append(self.rank_list[j])
            
#            logger.info('new_keyword_rank_list: ' + str(new_keyword_rank_list))
            self.rank_value.clear();
            self.rank_list.clear()
            self.rank_value = new_rank_value
            self.rank_list = new_keyword_rank_list
            
        else:
            if len(self.rank_value) < MAX_SELECT_STEPS:
                self.rank_value.append(rewardvalue)
                self.rank_list.append(str(keylist))
        
#        logger.info('last self.keyword_rank_list: ' + str(self.rank_list))


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

MAX_WORDSODE = 10000
MAX_SELECT_STEPS = 20
POPULARITY_BOUND = 1000000

evn_word = WordAgent('assert/keyword.xlsx','xlsx')
evn_word.openExcel()
evn_word.reset()

batch_size  = 32

n_features = evn_word.get_observation().shape[0]
n_actions = 1
a_bound = np.array([2.])
memory_size = 10000

logger.debug('n_features: ' + str(n_features))
logger.debug('n_actions: ' + str(n_actions))
logger.debug('a_bound: ' + str(a_bound))
###############################  training  ####################################


        

ddpg = DDPG4KeyWords(n_actions=n_actions,
        n_features=n_features,
        reward_decay=0.9,
        lr_a = 0.001,
        lr_c = 0.002,
        TAU = 0.01,
        output_graph=False,
        log_dir = 'WordGame/log/DDPG4KeyWords/',
        a_bound =a_bound,
        model_dir = 'WordGame/model_dir/DDPG4KeyWords/')
memory = Memory(memory_size=memory_size)
var = 3  # control exploration
t1 = time.time()
step = 0
rank = KeyWordRank()
for i in range(MAX_WORDSODE):
    s = evn_word.reset()
    logger.debug('env.reset() s: ' + str(s))
    ep_reward = 0
    print('env s:')
    print(s)
    for j in range(MAX_SELECT_STEPS):
        step+=1
        logger.debug('current step: ' + str(step))
        # Add exploration noise
        a = ddpg.choose_action(s)
        logger.debug('action choose: ' + str(a))
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        logger.debug('var action choose: ' + str(a))
        s_, r, done, info = evn_word.step(a)

        memory.store_transition(s, a, r / POPULARITY_BOUND, s_)
        logger.debug('memory.store_transition: ' + str(s) + ' ,' + str(a) + ' ,' + str(r/POPULARITY_BOUND) + ' ,' + str(s_))

        if step > memory_size:

                logger.debug('DDPG4KeyWords learning :-----------------' + str(step))
                var *= .9999995    # decay the action randomness
                logger.debug('learn var: ' + str(var))
                data = memory.sample(batch_size)
                ddpg.learn(data)
                logger.debug('end DDPG4KeyWords learning :-----------------')

        s = s_
        ep_reward += r
        if j == MAX_SELECT_STEPS-1:
           logger.info('result: ' + str(evn_word.result))
           result_keywords = evn_word.get_result_keywords()
           logger.info('Episode:' + str(i) + ' Reward: ' + str(ep_reward) + ' Result: ' + str(result_keywords))
           rank.checkrank(ep_reward,result_keywords)
           logger.info('current rank value: ')
           logger.info(rank.rank_value)
           logger.info('current rank words: ')
           for n in range(len(rank.rank_list)):
               logger.info(rank.rank_list[n])
           break
       
logger.info('Running time: ', time.time() - t1)
logger.info('Finally rank value: ')
logger.info(rank.rank_value)
logger.info('Finally rank words: ')
for n in range(len(rank.rank_list)):
    logger.info(rank.rank_list[n])