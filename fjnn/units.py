'''
Different types of units one needs to put together a DNN. Include neurons and losses.
'''

import tensorflow as tf
import numpy as np
from ._defaults import * #some globals

def l1_norm(T, name=None):
    '''l1 norm of the tensor'''
    return tf.reduce_sum( tf.abs(T), name=name)

def mse(y, yhat, w=None, name=None):
    '''Weighted MSE that deals with broadcasting gotchas
    (see http://stackoverflow.com/questions/16995071/numpy-array-that-is-n-1-and-n)
    '''
    r = tf.reshape( y, (-1,) ) - tf.reshape( yhat, (-1,) )
    if w is None:
        return tf.reduce_mean(r**2, name=name)
    w = tf.reshape(w, (-1,))
    return tf.div( tf.reduce_sum( w * r**2 ), tf.reduce_sum(w), name=name )

def crelu(T, name='crelu'):
    ''' the concatenated relu unit = but better than the TF implementation
    https://www.tensorflow.org/api_docs/python/nn/activation_functions_#crelu
    :param T: tensor to crelu
    :return: The crelu activation of T concated along the dimension 1
    '''
    return tf.concat(1, [tf.nn.relu(T), tf.nn.relu(-T)], name=name )
    
def dense(name,  #name of the unit - or scope variable
          input, nout, bias=False, activation=tf.identity,
          W_init=TRN_INITIALIZER, b_init=RU_INITIALIZER,  # initializers
          W_l2=None, b_l2=None, W_l1=None, b_l1=None,  # l2 and l1 regularization penalties
          W_c=None, b_c=None,  # weight and bias constraints (TBD)
          dtype=DTYPE):
    '''
    Create a dense layer, along with regularization and summarizers.
    Reshape the input if needed.
    Example:
    with tf.variable_scope( scope, reuse=<True|False>, initializer=<default_initializer>):
        output = fnu.dense(X, 10, bias=True, activation=tf.relu,
            W_init=tf.random_uniform_initializer(0.1,0.9), b_init=tf.constant_initializer(0.4),
            W_l2 = 0.1, summaries=['activation'])
    '''
    # check the input for correct shape
    input_shape = input.get_shape().as_list()
    assert len(input_shape) >= 2, 'incorrectly shaped input'
    if len(input_shape) > 2:
        # reshape the input if needed
        input = tf.reshape(input, [-1, np.prod(input_shape[1:])] )
        input_shape = input.get_shape().as_list()
    # create the dense layer
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=(input_shape[-1], nout), initializer=W_init, dtype=dtype);
        b = tf.get_variable(name='b', shape=(nout,), initializer=b_init, dtype=dtype) \
            if bias else tf.constant(0, dtype=dtype, name='b')
        output = activation( tf.matmul(input, W) + b, name='output' )
    tf.add_to_collection(WEIGHTS, W)
    tf.add_to_collection(BIASES, b)
    tf.add_to_collection(ACTIVATIONS, output)
    # Add regularizations
    if W_l2 is not None:
        tf.add_to_collection(REGULARIZERS, tf.mul( tf.nn.l2_loss(W), W_l2, name='W_l2' ));
    if W_l1 is not None:
        tf.add_to_collection(REGULARIZERS, tf.mul( l1_norm(W), W_l1, name='W_l1' ));
    if b_l2 is not None:
        tf.add_to_collection(REGULARIZERS, tf.mul( tf.nn.l2_loss(b), b_l2, name='b_l2' ));
    if b_l1 is not None:
        tf.add_to_collection(REGULARIZERS, tf.mul( l1_norm(b), b_l1, name='b_l1' ));
    return output