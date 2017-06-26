#!/usr/bin/env python3
"""
Train the eccentricity model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
np.random.seed(seed=1)

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import convert_to_records as records
import ecc
import json


# from tensorflow.models.image.cifar10 import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 20,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
def train_dir():
  return os.path.join(FLAGS.train_dir, 'trained', FLAGS.model_name, 'tffiles')

def train():
  """Train eccentricity mdoel for a number of steps."""
  json.dumps(FLAGS.__dict__, os.path.join(FLAGS.train_dir, 
    'pm{}_lr{:.0e}_c{}'.format(FLAGS.pm, FLAGS.learning_rate, FLAGS.chevron), 
    'settings.json'), 
    ensure_ascii=True)
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = ecc.inputs('train', FLAGS.batch_size, FLAGS.num_epochs)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = ecc.inference(images)

    # Calculate loss.
    loss = ecc.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = ecc.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.train.SummaryWriter(train_dir(), sess.graph)

    print('Settings used are:')
    f = FLAGS.__dict__['__flags']
    for key in sorted(f.keys()):
      print('{} : {}, type {}'.format(key, f[key], type(f[key])))

    # debug_print = tf.Print(images, [images, tf.shape(images), tf.reduce_max(images), tf.reduce_min(images)], message="Images at runtime are")
    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        # sess.run(debug_print)
        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

        if step % 100 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0:
          checkpoint_path = os.path.join(train_dir(), 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

        step += 1
    
    except tf.errors.OutOfRangeError:
      checkpoint_path = os.path.join(train_dir(), 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step=step)

      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def main(argv=None):  
  records.maybe_download_and_preprocess_training()
  if tf.gfile.Exists(train_dir()):
    tf.gfile.DeleteRecursively(train_dir())
  tf.gfile.MakeDirs(train_dir())
  train()


if __name__ == '__main__':
  tf.app.run()
