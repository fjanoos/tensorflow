'''
my implementation of the code in tensorflow/models/images/cifar10/cifar10.py and cifar10_train.py

define this env var if you're not getting memory on GPU0
>export CUDA_VISIBLE_DEVICES=1,2,3
'''

import tensorflow as tf
from pylab import *
import os.path, time, sys, re
import numpy as np
from datetime import datetime
from tensorflow.models.image.cifar10 import cifar10, cifar10_input

data_dir = '/home/ubuntu/fj/data/cifar10/'
cifar10.FLAGS.data_dir = data_dir;

# -- training constants --
train_dir = data_dir +'/training'
use_fp16 = False;
batch_size = 128;
log_device_placement = False; #wether to log device placement
max_steps = 1000000

# -- dataset constants --
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# -- Constants describing the training process --
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# -- evaluation constants --
eval_dir  = data_dir +'/eval'
eval_data = 'test' #Either 'test' or 'train_eval'.
eval_interval_secs = 60 ; #How often to run the eval
num_examples = 10000;  #Number of examples to run
run_once = False #Whether to run eval only once.

def _variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory. Place the variables on CPU memory and data on GPU memory
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        # get an existing variable with this name, or create a new one.
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized Variable with weight decay. Variable initialized with truncated normal
    """
    dtype = tf.float16 if use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        # store the weight decay term in the default graph.
        tf.add_to_collection('losses', weight_decay)
        # to inpsect the losses collection
        #   tf.get_default_graph().get_collection('losses')[0]
    return var

def _activation_summary(x):z
    """
    Helper to create summaries for activations.
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training session for clariy.
    tensor_name = re.sub('tower_[0-9]*/', '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _loss_summaries(total_loss):
    """
    Generates moving average for all losses and associated summaries for visualizing the performance of the network.
    """
    #EMA summarizer
    averager = tf.train.ExponentialMovingAverage(0.9, name='avg')
    #The set of all the weight decay losses.
    losses = tf.get_collection('losses')
    #apply() creates shadow variables that maintain moving averages and returns an op that does the ema
    #   shadow_variable = decay * shadow_variable + (1 - decay) * variable
    loss_averages_op = averager.apply( losses + [total_loss])
    # Attach a scalar summary to all individual losses and the total loss; do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        # average() returns the shadow variable for a given variable
        tf.scalar_summary(l.op.name +' (ema)', averager.average(l))
    return loss_averages_op

def inference(images):
    """
    Build the CIFAR-10 model.
    :param images: tensor of batch_size x width x height x channels size
    """
    # We instantiate all variables using tf.get_variable() instead of tf.Variable() to force cpu:0 placement.
    # This is in order to share variables across multiple GPU training runs. If we only ran this model on a single GPU,
    # we could simplify this function by replacing all instances of tf.get_variable() with tf.Variable().
    # -- conv1 --
    with tf.variable_scope('conv1') as scope:
        # 64-channels of 5x5x3 convolution kernels.
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        # stride = 1, i.e no downsampling.
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        # Special case of tf.add where bias is restricted to 1-D. Unlike tf.add, the type of bias may differ from value
        bias = tf.nn.bias_add(conv, biases)
        # shape is [batch, height, width, 64]
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
    # -- pool1 : pooling of 3x3 for each channel, with downsampling factor of 2 --
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    # -- norm1: local-response normalization (see http://goo.gl/MUclw8) --
    # sqr_sum[a, b, c, d] =  sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    # output = input / (bias + alpha * sqr_sum)**beta
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    # -- conv2 --
    with tf.variable_scope('conv2') as scope:
        # 64channels of 5x5x64 convolution kernels
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        # no downsampling. shape=[batch, height, width, 64]
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)
    # -- norm2 --
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # -- pool2: 3x3 pooling and downsample by 2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # -- local3: fully connected --
    with tf.variable_scope('local3') as scope:
        # Reshape the tensor - so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [batch_size, -1])
        dim = reshape.get_shape()[1].value #should be width/2/2 x height/2/2 x 64
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        # batch_size x 384
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)
    # --  local4 --
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        # batch_size x 192
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)
    # softmax, i.e. softmax(WX + b) -> still in logits. exp is not applied.
    with tf.variable_scope('softmax_logits') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        # shape = batch_size x NUM_CLASSES
        softmax_logits = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_logits)
    return softmax_logits

def loss(logits, labels):
    """
    Add L2Loss to all the trainable variables.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    #converts labels into 1-hot vectors. ent[x,y] = \sum_i y_i logit_xi - log(\sum_k logit_xk)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    # reduce along sample dimension
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n( tf.get_collection('losses'), name='total_loss' ) #add list of tensors element wise

def distorted_inputs():
    # what the fuck is going on in this function ??
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir+'/cifar-10-batches-bin/', batch_size=batch_size)
    if use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def train(total_loss, global_step):
    """
    Create an optimizer and apply to all trainable variables.
    Maintain EMA for all trainable variables.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    # Decay the learning rate exponentially based on the number of steps.
    # decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step,
                                    decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
    tf.scalar_summary('learning_rate', lr)
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _loss_summaries(total_loss)
    #Ensure that the steps inside the context will run only after argument ops are run.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        # Compute gradients of loss for the variables in GraphKey.TRAINABLE_VARIABLES.
        grads = opt.compute_gradients( total_loss )
    # Apply (scaled) gradients to variables. Update global_step.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    # Track the moving averages of all trainable variables.  The second parameter tweaks decay rate dynamically.
    # If passed, the actual decay rate used is: min(decay, (1 + num_updates) / (10 + num_updates))
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply( tf.trainable_variables() )
    # enforce control dependency on these two ops
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        # simply a placeholder to apply gradients and track variable EMAs
        train_op = tf.no_op(name='train')
    return train_op

def eval_once(saver, summary_writer, top_k_op, summary_op):
    """
    Evaluate the latest set of parameters
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        # read the trained model checkpoint
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores EMA variables into the session from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        # create a coordinator to manage the queue runners
        coord = tf.train.Coordinator()
        try:
            threads = []
            # Start the queue runners - for the input reader
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            num_iter = int(math.ceil(num_examples / batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1
            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

if __name__ == '__main__':
    import shutil
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',"--train", help="Run training phase", action='store_true')
    parser.add_argument('-e', "--eval", help="Run evaluation phase", action='store_true')
    args = parser.parse_args()

    # -- training phase --
    if args.train:
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir)
        # download the data - in case.
        cifar10.maybe_download_and_extract();
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False)
            # Get images and labels for CIFAR-10.
            images, labels = distorted_inputs()
            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = inference(images);
            # Calculate cross entropy loss.
            total_loss = loss(logits, labels)
            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op = train(total_loss, global_step)
            # Create a saver.
            saver = tf.train.Saver( tf.all_variables() )
            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()
            # Start running operations on the Graph.
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement))
            # Initialization operation
            sess.run(tf.initialize_all_variables())
            # create a coordinator to manage the threads
            coord = tf.train.Coordinator()
            # Start the queue runners (created in tf.train.shuffle_batch)-> to read the input
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # Logging
            summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
            try:
                for step in range(max_steps):
                    # has a thread signalled stopping ?
                    if coord.should_stop():
                        break;
                    start_time = time.time()
                    _, loss_value = sess.run([train_op, total_loss])
                    duration = time.time() - start_time
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    if step % 10 == 0:
                        num_examples_per_step = batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                        print (format_str%(datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                    if step % 100 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                    # Save the model checkpoint periodically.
                    if step % 1000 == 0 or (step + 1) == max_steps:
                        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            # Wait for threads to finish.
            coord.join(threads)
            sess.close()

    # -- evaluation phase : Eval CIFAR-10 for a number of steps --
    if args.eval:
        with tf.Graph().as_default() as g:
            # Get images and labels for CIFAR-10.
            images, labels = cifar10.inputs(eval_data=eval_data == 'test')
            # Build a Graph that computes the logits predictions from the inference model.
            logits = inference(images)
            # Calculate predictions.
            top_k_op = tf.nn.in_top_k(logits, labels, 1)
            averager = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
            # variables_to_restore(): returns the EMA shadow variable if the variable has a EMA, otherwise the variable.
            # args: moving_avg_variables: a list of variables whose the moving variable is to be
            #       restored. If None, it will default to tf.moving_average_variables() + tf.trainable_variables()
            variables_to_restore = averager.variables_to_restore()
            # create a saver for all the EMA variables to restore
            saver = tf.train.Saver(variables_to_restore)
            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(eval_dir, g)
            while True:
                eval_once(saver, summary_writer, top_k_op, summary_op)
                if run_once:
                      break
                time.sleep(eval_interval_secs)

