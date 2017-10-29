
from pylab import *
from keras import backend as KB
import keras as K
from pandas import *

import tensorflow as tf
sess = tf.InteractiveSession()


W_hh = tf.Variable( tf.zeros([10, 784]) )
b_hh = tf.Variable( tf.zeros([10, 1] ) )
h = KB.placeholder( shape=[784,1], dtype=W_hh.dtype )
#h = tf.Variable( tf.zeros([784, 1] ) )
oh = KB.tanh( KB.dot(W_hh, h) + b_hh)

sess.run(tf.initialize_all_variables())


class RNN:
    def __init__(self):
        self.yo = 'empty'
