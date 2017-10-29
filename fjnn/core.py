'''
Core operations for deep neural networks

@author fj
@date 2016-11-26
'''

import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
import re
from ._defaults import * #some globals

def summarize_tensor(T, summary='variable'):
    '''
    Add a summary for the given tensor. If summary='gradient', expects (grad, var) tuple
    Based on tf.contrib.layers.summaries.py
    :param summary: 'activation', 'variable', 'gradient' or 'training'
    return summary op
    '''
    var, name = (T[0], T[1].op.name) if summary == 'gradient' else (T, T.op.name)
    name = re.sub('tower_[0-9]*/', '', name) #any ohter filters here
    with tf.name_scope('summaries'):
        if var is None:
            return tf.no_op(name=name)
        # for scalars use a scalar summary
        if var.get_shape().ndims == 0:
            return tf.scalar_summary('{}/{}/scalar'.format(name, summary), var)
        # we may land here if rank is still unknown
        return (
            tf.histogram_summary( '{}/{}/histogram'.format(name, summary), var ),
            tf.scalar_summary( '{}/{}/norm'.format(name, summary), tf.nn.l2_loss(var) ),
        )

def summarize_tensors(tensors, summary):
    '''Summarize a set of tensors'''
    return [summarize_tensor(T, summary) for T in tensors]
    

def summarize_losses(losses=[], loss_ema_decay=LOSS_EMA_DECAY):
    '''
    create averaged summariziers for all losses and regularizers
    :param loss_ema_decay: ema for tracking loss variable
    '''
    with tf.name_scope('losses'):
        all_losses = set( tf.get_collection(REGULARIZERS) + tf.get_collection(LOSSES) + losses )
        # track an ema of all the losses
        loss_ema = tf.train.ExponentialMovingAverage(loss_ema_decay, name='loss_ema')
        loss_ema_op = loss_ema.apply( all_losses )
        # Attach a scalar summary to all losses and the ema version of the losses.
        for l in losses:
            tf.scalar_summary( l.op.name +' (raw)', l )
            # average() returns the shadow variable for a given variable
            tf.scalar_summary( l.op.name +' (ema)', loss_ema.average(l) )
    return loss_ema_op


def minimize(opt,               #optimizer instance
             trg_loss,          #optimization target
             global_step=None,  #global optimization counter
             variable_set=None, #default tf.trainable_variables()
             gradient_noise_scale=None, #add gradient noise
             gradient_multiplier=None,
             gradient_clipper=None,
             #--some other tf.train.Optimizer params --
             colocate_gradients_with_ops=False, #If True, try colocating gradients with op.
             gate_gradients=tf.train.Optimizer.GATE_OP
             ):
    assert trg_loss.get_shape().ndims == 0, 'training cost mis-specified'
    # get global step for this graph
    global_step = contrib_framework.assert_or_get_global_step(global_step_tensor=global_step)
    # get variables in GraphKey.TRAINABLE_VARIABLES
    variable_set = set( tf.trainable_variables() ) if variable_set is None else variable_set
    grads_and_vars = opt.compute_gradients( trg_loss, var_list=variable_set, gate_gradients=gate_gradients)
    # -- add gradient noise ---
    noisy_grads_and_vars = []
    if gradient_noise_scale is not None:
        for g, v in grads_and_vars:
            if g is None:
                noisy_grads_and_vars.append( (None, v) )
            if isinstance(g, tf.IndexedSlices):
                _shape = g.dense_shape;
            else:
                _shape = g.get_shape();
            noise = tf.truncated_normal(_shape) * gradient_noise_scale
            noisy_grads_and_vars.append( (g+ noise, v) )
        grads_and_vars = noisy_grads_and_vars
    # -- add summarizers for the gradients --
    summarize_tensors(grads_and_vars, 'gradient')
    # summarize total gradient norm
    summarize_tensor( tf.global_norm( list(zip(*grads_and_vars))[0], name='gradients') )
    # apply the descent step
    descent_op = opt.apply_gradients( grads_and_vars, global_step=global_step, name='descent_op' )
    return descent_op;
        

def ema_variables(global_step, variable_set=None, var_ema_decay=VARIABLE_EMA_DECAY):
    '''
    Track the moving averages of all trainable variables.  The second parameter tweaks decay rate dynamically.
    If passed, the actual decay rate used is: min(decay, (1 + global_step) / (10 + global_step))
    '''
    global_step = contrib_framework.assert_or_get_global_step(global_step_tensor=global_step)
    variable_set = set( tf.trainable_variables() ) if variable_set is None else variable_set
    variable_averager = tf.train.ExponentialMovingAverage(var_ema_decay, global_step)
    return variable_averager.apply( variable_set )
