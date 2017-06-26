"""Builds the eccentricity model.
Based on the tutorial for the CIFAR-10 model in Tensorflow.
http://tensorflow.org/tutorials/deep_cnn/

Relevant comments from that tutorial have been kept, others are added from me.

Summary of available functions:


 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

# from tensorflow.models.image.cifar10 import cifar10_input
import convert_to_records as records
import numpy as np
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
tf.app.flags.DEFINE_string('pm', '66661', 'pooling scheme across scales.  Each number specifies the number of scales remaining at each layer. The first number has to be the same as used in --num_scales.')
tf.app.flags.DEFINE_integer('conv_kernel', 5, 'Size of convolutional kernel')
tf.app.flags.DEFINE_integer('pool_kernel', 3, 'Size of spatial pooling kernel')
tf.app.flags.DEFINE_integer('feats_per_layer', 32, 'Number of feature channels at each layer')
tf.app.flags.DEFINE_boolean('total_pool', True, 'If true, pool all feature maps to 1x1 size in final layer')
tf.app.flags.DEFINE_integer('pool_stride', '1', 'If 2, we get progressive pooling - with overlap pooling, AlexNet style')


TRAIN_FILE = 'train_{}.tfrecords'.format(records.tfrecord_name())
VALIDATION_FILE = 'validation_{}.tfrecords'.format(records.tfrecord_name())

TEST_FILE = 'test_{}.tfrecords'.format(records.tfrecord_name())


def NUM_CLASSES():
  return 10 if FLAGS.parity == 'none' else 5

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  if FLAGS.contrast_norm == 'areafactor':
      image = tf.decode_raw(features['image_raw'], tf.float32)
  else:
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32) * (1. / 255)
 
  image.set_shape(np.prod([FLAGS.num_scales, FLAGS.crop_size, FLAGS.crop_size]))
  image = tf.reshape(image, [FLAGS.num_scales, FLAGS.crop_size, FLAGS.crop_size, 1])
  image = image - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label


def inputs(name, batch_size, num_epochs):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and test (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, FLAGS.num_scales, FLAGS.crop_size, FLAGS.crop_size]
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, NUM_CLASSES()).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(FLAGS.train_dir, 'data',
                          '{}_{}.tfrecords'.format(name, records.tfrecord_name()))

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=8,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv_scale(x, W):
  return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='VALID')


def inference(x):
  """ Creates a model with pooling across space and scales.
  Always we have a conv-relu-spatial_pool-scale_pool x N layers structure
  with one fully connected layer on top.
  """
      
  if '-' in FLAGS.pm:
    FLAGS.pm= FLAGS.pm.split('-')

  num_layers = len(FLAGS.pm) - 1
  print(num_layers)
  for l in range(num_layers):
    with tf.variable_scope('layer{}'.format(l)):
      with tf.variable_scope('conv'):
        if l == 0:
          bottom = x
          W = weight_variable([1, FLAGS.conv_kernel, FLAGS.conv_kernel, 1, FLAGS.feats_per_layer])
        else:
          if out.get_shape()[2] < FLAGS.conv_kernel:
            bottom = out # l (not l + 1) because from previous layer
            W = weight_variable([1, 1, 1, FLAGS.feats_per_layer, FLAGS.feats_per_layer])
          else:
            bottom = out # l (not l + 1) because from previous layer
            W = weight_variable([1, FLAGS.conv_kernel, FLAGS.conv_kernel, FLAGS.feats_per_layer, FLAGS.feats_per_layer])

        b = bias_variable([FLAGS.feats_per_layer])
        Wx_b = tf.nn.conv3d(bottom, W, strides=[1,1,1,1,1], padding='VALID') + b
        out = tf.nn.relu(Wx_b)
        shape = out.get_shape()
        print('conv{}'.format(l+1))
        print('\t{} --> {}'.format(bottom.name, out.name))
        print('\t{} --> {}'.format(bottom.get_shape(), out.get_shape()))
      with tf.variable_scope('pool'):
        bottom = out
        if l == num_layers - 1 and FLAGS.total_pool:
          kernel_size = bottom.get_shape()[2]
          out = tf.nn.max_pool3d(bottom, ksize=[1,1, kernel_size, kernel_size,1], strides=[1,1,1,1,1], padding='VALID')
        else:
          out = tf.nn.max_pool3d(bottom, ksize=[1,1, FLAGS.pool_kernel, FLAGS.pool_kernel,1], strides=[1,1,FLAGS.pool_stride,FLAGS.pool_stride,1], padding='VALID')
        shape = out.get_shape()
        print('pool{}'.format(l + 1))
        print('\t{} --> {}'.format(bottom.name, out.name))
        print('\t{} --> {}'.format(bottom.get_shape(), out.get_shape()))
      with tf.variable_scope('scale'):
        bottom = out
        if FLAGS.pm[l + 1]  == FLAGS.pm[l]:
          kernel_size = 1 # useless 1x1 pooling
        elif int(FLAGS.pm[l + 1]) < int(FLAGS.pm[l]):
          num_scales_prev = int(FLAGS.pm[l])
          num_scales_current = int(FLAGS.pm[l + 1])
          kernel_size = (num_scales_prev - num_scales_current) + 1
        else:
          raise ValueError('Number of scales must stay constant or decrease, got {}'.format(FLAGS.pm))
        out = tf.nn.max_pool3d(bottom, ksize=[1,kernel_size,1,1,1], strides=[1,1,1,1,1], padding='VALID')
        shape = out.get_shape()
        print('scale{}'.format(l + 1))
        print('\t{} --> {}'.format(bottom.name, out.name))
        print('\t{} --> {}'.format(bottom.get_shape(), out.get_shape()))

  with tf.variable_scope('fully_connected'):
    bottom = out
    bottom_shape = bottom.get_shape().as_list()
    reshape = tf.reshape(
        bottom,
        [-1, bottom_shape[1] * bottom_shape[2] * bottom_shape[3] * bottom_shape[4]])

    W_fc1 = weight_variable([bottom_shape[1] * bottom_shape[2] * bottom_shape[3] * bottom_shape[4], NUM_CLASSES()])
    b_fc1 = bias_variable([NUM_CLASSES()])
    out = tf.matmul(reshape, W_fc1) + b_fc1
    print('fc')
    print('\t{} --> {}'.format(bottom.name, out.name))
    print('\t{} --> {}'.format(bottom.get_shape(), out.get_shape()))
    if isinstance(FLAGS.pm, list):
      FLAGS.pm = '-'.join(FLAGS.pm)
    return out

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, FLAGS.NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def train(loss, global_step):
  """Train eccentricity model.

  Create an optimizer and apply to all trainable variables. 

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
   # Compute gradients.

  tf.scalar_summary(loss.op.name, loss)

  optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

  return train_op
