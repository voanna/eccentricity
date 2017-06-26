import os
import stat

# SETTING
EXPERIMENT = 'demo-experiment'


os.makedirs('experiments/{}/gen/scripts'.format(EXPERIMENT), exist_ok=True)
os.makedirs('experiments/{}/gen/trainlogs'.format(EXPERIMENT), exist_ok=True)
os.makedirs('experiments/{}/gen/trained'.format(EXPERIMENT), exist_ok=True)

train_all_sh = os.path.join('experiments/{}/gen/scripts/train_all.sh'.format(EXPERIMENT))
with open(train_all_sh, 'w') as g:
  
  # SETTING Settings common to all models to train for this experiment:
  pm = '11-1-1-1-1'
  total_pool = True

  g.write('#!/bin/sh\n')
  # SETTING Settings to change among different models:
  for lr in [0.1, 0.01, 0.001]:
    for contrast_norm in ['areafactor',  'None']:

      # SETTING
      model_name = '{}_total_pool_contrast_{}_lr{}'.format(pm, contrast_norm, lr)

      # SETTING
      pycmd = ('src/python/ecc/ecc_train.py '
        '--parity=even '
        '--learning_rate={} '
        '--pm={} '
        '--model_name={} '
        '--total_pool={} '
        '--train_dir={} '
        '--num_scales={} '
        '--contrast_norm={} '
        '--random_shifts '.format( 
          lr, 
          pm, 
          model_name,
          total_pool,
          os.path.join('experiments', EXPERIMENT, 'gen'),
          pm.split('-')[0], # num scales
          contrast_norm))
      slurmcmd = 'sbatch -n 8 --gres=gpu:titan-x:1 --mem=8000 --time=06:00:00'
      log= ' -o experiments/{}/gen/trainlogs/{}.log '.format(EXPERIMENT, model_name)
      shname = os.path.join('experiments/{}/gen/scripts'.format(EXPERIMENT), 'train_{}.sh'.format(model_name))
      with open(shname, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write(pycmd)
      st = os.stat(shname)
      os.chmod(shname, st.st_mode | stat.S_IEXEC)
      
      g.write(slurmcmd + log + shname + '\n')

    # wait until the jobs with 0.1 learning rate have already 
    # created the needed tffiles, so we don't have parallel processes 
    # writing to the same file and corrupting it.
    # here we just check that all jobs finished before submitting the rest
    if lr == 0.1:
      g.write('echo  Sleeping until first tffiles are created...\n')
      g.write('while (( $(squeue -u $USER | wc -l) > 1 )); do sleep 1m; done\n')
      g.write('echo Done sleeping, submitting jobs...\n')

      # print(slurmcmd + log + shname)

st = os.stat(train_all_sh)
os.chmod(train_all_sh, st.st_mode | stat.S_IEXEC)
