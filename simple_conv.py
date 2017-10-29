'''
Example of simple convolutional network in tensorflow

@version 1
@author fj
'''

import tensorflow as tf
from pylab import *
import os.path, time, sys, re
import numpy as np
from datetime import datetime
from pandas import *

with tf.Graph().as_default() as g:
    y = tf.placeholder(tf.float32, shape=None)
    for p in range(P):
        xvars.append( tf.placeholder(tf.float32, shape=None, name='x_%d'%p) )