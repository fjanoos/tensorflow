'''
some defaults

@date 2016-11-26
@author firdaus
'''
import tensorflow as tf

#general constants
DTYPE = tf.float32
RN_INITIALIZER = tf.random_normal_initializer(stddev=1.0)
TRN_INITIALIZER = tf.truncated_normal_initializer(stddev=1.0)
RU_INITIALIZER = tf.random_uniform_initializer(0.01, 0.99)
LOSS_EMA_DECAY = 0.9

# graph collections (tf.GraphKeys)
WEIGHTS = tf.GraphKeys.WEIGHTS
BIASES = tf.GraphKeys.BIASES
ACTIVATIONS = tf.GraphKeys.ACTIVATIONS
TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
MODEL_VARIABLES = tf.GraphKeys.MODEL_VARIABLES
REGULARIZERS = tf.GraphKeys.REGULARIZATION_LOSSES
LOSSES = 'all losses'
LEARNING_RATES = 'learning rates'

#optimization related stuff
N_EPOCHS = 50
BATCH_SIZE = 128
VARIABLE_EMA_DECAY = 0.9
INITIAL_LEARNING_RATE = 0.1      # Learning rate decay factor.
LEARNING_RATE_DECAY_FACTOR = 0.1 # Initial learning rate.
NUM_STEPS_PER_DECAY = 350        # optimization after which learning rate decays.
