#!/usr/bin/env python3

import ecc_eval
import ecc
import time
import itertools
import numpy as np 
import tensorflow as tf
import os

from run_crowding_acc import all_jobs, best_models, EXPERIMENT

start_time = time.time()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('job_id', -1, 'Job ID in grid')
tf.app.flags.DEFINE_integer('model_id', -1, 'Model ID')


def main(argv=None):
  job_id = FLAGS.job_id
  job = all_jobs[job_id]

  model_id = FLAGS.model_id
  model =  best_models()[model_id]

  print(job)
  print(model)

  flanker_type, target_ecc, flanker_ecc = job
  
  # SETTING
  total_pool = True
  pm = '11-1-1-1-1'

  # SETTING
  pm, contrast_norm, lr = model
  model_name = '{}_total_pool_contrast_{}_lr{}'.format(pm, contrast_norm, lr)

  precision_dir = 'experiments/{}/gen/precision_{}/'.format(EXPERIMENT, model_name)

  # SETTING settings from the trained model
  ecc_eval.FLAGS.train_dir = os.path.join('experiments', EXPERIMENT, 'gen')
  ecc_eval.FLAGS.parity = 'even'
  ecc_eval.FLAGS.learning_rate = lr
  ecc_eval.FLAGS.pm = pm
  ecc_eval.FLAGS.total_pool = total_pool
  ecc_eval.FLAGS.model_name = model_name
  ecc_eval.FLAGS.num_scales = int(pm.split('-')[0])
  ecc_eval.FLAGS.contrast_norm = contrast_norm


  # SETTING settings for the dataset used for testing
  ecc_eval.FLAGS.eval_name = '{}_ftype_{}_te_{}_fe_{}'.format(model_name, flanker_type, target_ecc, flanker_ecc)
  ecc_eval.FLAGS.random_shifts = False
  ecc_eval.FLAGS.flanker_dset = 'MNIST' 

  # if using notMNIST for example, need to also set ecc_eval.FLAGS.flanker_height = 85 (see README for explanation)

  # here we just select the datapoint (combination of target & flanker ecc and number of flankers) 
  # we want to evaluate
  ecc_eval.FLAGS.target_ecc = target_ecc
  ecc_eval.FLAGS.flanker_ecc = flanker_ecc
  ecc_eval.FLAGS.flanker_type = flanker_type

  print('Settings used are:')
  f = FLAGS.__dict__['__flags']
  for key in sorted(f.keys()):
    print('{} : {}'.format(key, f[key]))

  precision_fname = os.path.join(precision_dir, str(job_id) + '.out')

  if os.path.exists(precision_fname):
    print('Precision already computed.')
  else:
    precision = ecc_eval.main()
    print('Done in {} sec'.format(time.time()- start_time))

    with open(precision_fname, 'w') as f:
      f.write(str(precision))

if __name__ == '__main__':
  tf.app.run()
