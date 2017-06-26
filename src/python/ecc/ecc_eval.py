"""Evaluate eccentricity model


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
np.random.seed(seed=1)

from six.moves import xrange 
import tensorflow as tf
import convert_to_records as records
import ecc
import json 


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('eval_iter', 4180,
                            "Iteration at which to evaluate model")

tf.app.flags.DEFINE_string('eval_name', 'eval_name', 'Directory in which to save evaluation model')



def eval_dir():
  return os.path.join(FLAGS.train_dir, 'eval', FLAGS.eval_name)

def evaluate():
  """Eval eccentricity model"""

  # dump settings to a JSON
  settings = FLAGS.__dict__
  for key in settings['__flags'].keys():
    if isinstance(settings['__flags'][key], np.floating):
      settings['__flags'][key] = float(settings['__flags'][key])
    elif isinstance(settings['__flags'][key], np.integer):
      settings['__flags'][key] = int(settings['__flags'][key])

  json.dumps(settings, os.path.join(eval_dir(), 'settings.json'), 
    ensure_ascii=True, indent=4, sort_keys=True)  

  print('Settings used are:')
  f = FLAGS.__dict__['__flags']
  for key in sorted(f.keys()):
    print('{} : {}'.format(key, f[key]))
    
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = ecc.inputs('test', FLAGS.batch_size, 1)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = ecc.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Create a saver.
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))

    saver.restore(sess, 
      os.path.join(FLAGS.train_dir, 'trained', FLAGS.model_name, 
        'tffiles', 'model.ckpt-{}'.format(FLAGS.eval_iter)))
    
    for var in tf.all_variables():
      try:
        sess.run(var)
      except tf.errors.FailedPreconditionError:
        print('*'*70)
        print(var)

    sess.run(tf.initialize_local_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    summary_writer = tf.train.SummaryWriter(
      os.path.join(eval_dir(), 'tffiles'), sess.graph)

    try:
      step = 0
      true_count = 0
      start_time = time.time()

      while not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        
        step += 1
 
    except tf.errors.OutOfRangeError:
      print('Done evaluating for 1 epoch, %d steps.' % (step))
      # Compute precision @ 1.
      total_sample_count = step * FLAGS.batch_size
      precision = true_count / total_sample_count
      duration = time.time() - start_time

      print('Duration %s: precision @ 1 = %.3f' % (duration, precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, 4180)
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    return precision


def main(argv=None): 
  records.maybe_download_and_preprocess_evaluation()
  if tf.gfile.Exists(eval_dir()):
    tf.gfile.DeleteRecursively(eval_dir())
  tf.gfile.MakeDirs(eval_dir())
  precision = evaluate()
  return precision 

if __name__ == '__main__':
  tf.app.run()
