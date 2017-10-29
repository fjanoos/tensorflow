'''
Test convolutions using keras + tensorflow
'''

import sys, os
from keras.engine.topology import Layer
import tensorflow as tf
import keras as K
import keras.backend as KB
from numba import jit
import numpy as np
from pylab import *
from pandas import *
from statsmodels.api import OLS
import seaborn as sns
sns.set_style('whitegrid')
import fjnn


# -- globals --
_dtype = tf.float32;
_batch_size = 1024;
_nb_epoch = 500;
train_dir = '/home/ubuntu/deep-learning/data/basic_conv'
os.makedirs(train_dir, exist_ok=True)

@jit(nopython=True, nogil=True)
def causal_conv(x, h):
    ''' Causal convolution.
        Note np.convolve is a non-causal convolution
        :returns causal convoluation of same shape
    '''
    n = len(x);
    y = np.zeros( n );
    for t in range(n):
        if not np.isfinite(x[t]): continue
        for s in range( len(h) ):
            y[t] += x[t-s] * h[s]
    return y

@jit(nopython=True, nogil=True)
def hankel(x, w):
    '''
    Reshape the x vector into a causal hankel matrix of width w
    op[t] = [x_t ... x_{t-w}]
    '''
    n = len(x)
    op = np.zeros( (n, w))
    for t in range(n):
        for v in range(w):
            if t - v < 0 : break;
            op[t, v] = x[t - v]
    return op

def _gfunc(n, sigma):
    '''gaussian functions'''
    return exp( -pow(linspace(-3, 3, n), 2) / 2 / sigma**2 ) / sqrt(2*pi*sigma**2)

def build_hankel_tensors(df, P, w):
    ''' Tensorize the hankel matrices as num_examples x time  x feature  '''
    return np.transpose(
        np.array([ hankel(df['x_%d'%p].values, w) for p in range(P) ]), [1,0,2] )

def get_data(N, P, H, wf='rand'):
    '''
    Create the linear convolution dataset
    :param N: num samples
    :param P: num features
    :param H: convolution lenght
    :param wf: waverform
    :return:
    '''
    # the generating convolution kernels
    conv_kernels = [_gfunc(H, p) * cos(linspace(0, 2 * pi * (p - 1), H)) for p in range(1, P + 1)]
    # the input dataframe
    if wf == 'sin':
        df = DataFrame( {'x_%d'%p: sin( linspace(0, 2*pi*(3.33*p+1), N) )   for p in range(P)})
    elif wf == 'sq':
        df = DataFrame( {'x_%d'%p: hstack( [ hstack([ones(50)/(p+1), -ones(50)/(p+1)] )
                                        for _ in range(N//100) ] ) for p in range(P)} )
    elif wf == 'rand':
        df = DataFrame( {'x_%d'%p: randn(N) for p in range(P)})
    df['e'] = 0.1*randn( N )
    df['y'] = 0;
    for p in range(P):
        xc = causal_conv( df['x_%d'%p].values, conv_kernels[p] )
        df['y'] +=  xc
    df['yl'] = df['y'] + df['e']
    df['ynl'] = cos(df['y']) + df['e']*0.1
    return df, conv_kernels

def linear_deconv_keras(X, y, nb_epoch=_nb_epoch, batch_size=_batch_size):
    ''' linear deconvolution
        instrument with tensorflow summary objects
    '''
    x = K.layers.Input(X.shape)
    P, H = X.shape[1], X.shape[2]
    model = K.models.Sequential( [ K.layers.Reshape( (P*H, ), input_shape=[P, H] ) ] )
    model.add( K.layers.Dense(1) );
    model.compile(loss='mse', optimizer=K.optimizers.Nadam(), metrics=['mean_squared_error'] )
    model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
    return model

def linear_deconv_tf(X, y,  nb_epoch=_nb_epoch, batch_size=_batch_size):
    x = tf.placeholder( shape=X.shape)

def nonlinear_deconv(X, y, nb_epoch=_nb_epoch, batch_size=_batch_size):
    ''' non linear deconvolution '''
    x = K.layers.Input(X.shape)
    P, H = X.shape[1], X.shape[2]
    model = K.models.Sequential( [ K.layers.Reshape( (P*H, ), input_shape=[P, H] ) ] )
    model.add( K.layers.Dense(1) );
    return model(x)

def linear_test(N, P, H, nb_epoch, batch_size):
    '''  linear deconvolution with keras    '''
    df, conv_kernels = get_data(N, P, H)
    X = build_hankel_tensors(df, P, H)
    ols = OLS(
        endog=df['y'],
        exog=np.reshape(X, [-1,P*H] )
    ).fit()
    km = linear_deconv_keras(X, df['yl'].values, nb_epoch, batch_size)
    w = km.get_weights()[0]
    for p in range(P):
        subplot(221); plot( conv_kernels[p] )
        subplot(222); plot( df['x_%d'%p] )
    subplot(223); plot( df['yl'], '-', alpha=0.5)
    subplot(224); plot(ols.params.values, label='ols')
    subplot(224); plot(w, label='keras')
    subplot(224); gca().legend()
    plt.show()

def nonlinear_test(N, P, H, nb_epoch, batch_size):
    return;

if __name__ == '__main__':
    with tf.Graph().as_default() as g, tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
        saver = tf.train.Saver( tf.all_variables() )
        saver.save(sess, os.path.join(train_dir, 'model.ckpt') )
        if sys.argv[1] == 'linear':
            linear_test( *(int(a) for a in sys.argv[2:]) )

## --- GARBAGE --
class HankelConv(Layer):
    ''' Do a point in time convolution with Hankelized data matrix
        NOTE: This is totally not needed. A simple dense layer after rehaping the input into vector form
        should suffice.
    '''
    def __init__(self, output_dim, stdev=0.1, initializer='truncated_normal', **kwargs):
        self.output_dim = output_dim
        self.initializer = tf.truncated_normal_initializer(stddev=stdev, dtype=_dtype)
        self.stdev=0.1
        super(HankelConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.winlen = input_shape[2]
        self.P = input_shape[1]
        self.trainable_weights = [];
        # create one winlen x nfilters convolution matrix for each input feature p
        with tf.variable_scope('HankelConv'):
            for p in range(self.P):
                self.trainable_weights.append(
                    tf.get_variable(name='W_{}'.format(p), shape=[self.winlen, self.output_dim],
                                    dtype=_dtype, initializer=self.initializer)
                )
        self.built = True

    def call(self, x, mask=None):
        ''' Multiply the last dimension of x by the winlen x nfilters matrix for
            each input feature p
            :param x: [None, P, window] (hankelized )
            returns: tensor of shape [None, P, nfilters]
        '''
        y = [];
        with tf.variable_scope('HankelConv'):
            for p in range(self.P):
                # the batch_matmul returns a [None, output_dim] matrix (via broadcsting)
                # which is kind of strange. It squeezes out a dimension but the documentation doesnt mention that.
                # The reshape adds that dim back in
                y.append( tf.reshape( tf.batch_matmul(x[..., p, :], self.trainable_weights[p]),
                                      [-1, 1, self.output_dim] ) )
            #concat the results for each features to create [None, P, output_dim] tensor
            return tf.concat(1, y)

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 3, 'incorrect rank tensor'
        return (input_shape[0], input_shape[1], self.output_dim)

def test_hankel_deconv(df, P, H):
    ''' Test the HankelConv Layer
    '''
    X = build_hankel_tensors(df, P, H)
    x = tf.Variable(X, dtype=tf.float32)
    model = K.models.Sequential( [ HankelConv(1, input_shape=[X.shape[1], X.shape[2]] ) ] )
    print( model(x).get_shape() )
    model.add( K.layers.Reshape((P, )) )
    print( model(x).get_shape() )
    model.add( K.layers.Dense(1, trainable=False, weights=np.ones((P,1))) )
    print(model(x).get_shape())
    model.compile(loss='mse', optimizer=K.optimizers.Nadam(), metrics=['mean_squared_error'] )
    model.fit(X, df['yl'].values, nb_epoch=100, batch_size=1000 )
    return model;

