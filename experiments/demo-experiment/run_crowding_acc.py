import itertools
import numpy as np
import os
import stat
import re
import subprocess

# SETTING
EXPERIMENT = 'demo-experiment'

edge = 900
step = 60
flanker_eccs = np.arange(0, edge, 2*step)
target_eccs = np.arange(0, 840, step)
flanker_types = [0, 1, 2]
all_jobs = list(itertools.product(flanker_types, target_eccs, flanker_eccs))

def best_models(run_from_expt_dir=False):
  all_models = []

  # SETTING
  pm = '11-1-1-1-1'
  total_pool = True

  # SETTING
  for contrast_norm in ['areafactor', 'None']:
    best_loss = float('inf')
    best_lr = 0
    for lr in [0.1, 0.01, 0.001]:

      # SETTING
      model_name = '{}_total_pool_contrast_{}_lr{}'.format(pm, contrast_norm, lr)
      if run_from_expt_dir:
        logfile = 'gen/trainlogs/{}.log'.format(model_name)
      else:
        logfile = 'experiments/{}/gen/trainlogs/{}.log'.format(EXPERIMENT, model_name)
      output = subprocess.check_output("tail -2 {} | head -1".format(logfile), shell=True)
      m = re.search("step {}, loss = [0-9.]+".format(4170).encode(), output)
      if m is not None:
        loss = float(str(m.group()).split(" ")[-1][:-1])
        if loss <= best_loss:
          best_loss = loss
          best_lr = lr

    all_models.append([pm, contrast_norm, best_lr])
  return all_models


def main():
  count = 0
  eval_all_models_sh = os.path.join('experiments/{}/gen/scripts/eval_all.sh'.format(EXPERIMENT))

  with open(eval_all_models_sh, 'w') as h:
    h.write('#!/bin/sh\n')
    for j, model in enumerate(best_models()):
      pm, contrast_norm, lr = model
      model_name = '{}_total_pool_contrast_{}_lr{}'.format(pm, contrast_norm, lr)

      precision_dir = 'experiments/{}/gen/precision_{}/'.format(EXPERIMENT, model_name)
      evallog_dir = 'experiments/{}/gen/evallog_{}/'.format(EXPERIMENT, model_name)

      os.makedirs(precision_dir, exist_ok=True)
      os.makedirs(evallog_dir, exist_ok=True)

      eval_all_sh = os.path.join('experiments/{}/gen/scripts/eval_all_{}.sh'.format(EXPERIMENT, model_name))

      with open(eval_all_sh, 'w') as g:
        g.write('#!/bin/sh\n')
        for i in range(len(all_jobs)):
          count += 1
          shname = os.path.join('experiments/{}/gen/scripts'.format(EXPERIMENT), '{}_eval{}.sh'.format(model_name, i))
          
          pycmd = 'experiments/{}/eval_all.py --job_id={} --model_id={}'.format(EXPERIMENT, i, j)
          slurmcmd = 'sbatch -n 1 --gres=gpu:1 --mem=8000 --time=01:00:00'
          log= ' -o {}'.format(os.path.join(evallog_dir, str(i) + '.log'))

          with open(shname, 'w') as f:
            f.write('#!/bin/sh\n')
            f.write(pycmd)

          st = os.stat(shname)
          os.chmod(shname, st.st_mode | stat.S_IEXEC)
          
          g.write(slurmcmd + ' ' + log + ' ' + shname + '\n')
          if count % 200 == 0:
            g.write('echo  Sleeping until jobs finish...\n')
            g.write('while (( $(squeue -u $USER | wc -l) > 300 )); do sleep 1m; done\n')
            g.write('echo Done sleeping, submitting jobs...\n')
          # print(slurmcmd + ' ' + log + ' ' + shname)

      st = os.stat(eval_all_sh)
      os.chmod(eval_all_sh, st.st_mode | stat.S_IEXEC)

      h.write(eval_all_sh + '\n')

  st = os.stat(eval_all_models_sh)
  os.chmod(eval_all_models_sh, st.st_mode | stat.S_IEXEC)

if __name__ == '__main__':
  main()
