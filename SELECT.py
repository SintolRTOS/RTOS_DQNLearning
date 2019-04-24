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
print('init lib res sucessful!')

from Util import Memory,StateProcessor
from DDPG import DDPG
print('init ddpg sucessful!')
from ACNetwork import ACNetwork