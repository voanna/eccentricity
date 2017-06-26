# README
This code accompanies the paper "Do Deep Neural Networks Suffer from Crowding?" by Anna Volokitin, Gemma Roig and Tomaso Poggio [1].

The main purpose of this repository is to provide an implementation of the eccentricity-dependent model [3], as well as to show an example of our experiments carried in [1]. 
This code is inspired by the implementation described in  [2]. Yet, it is is not intended  to replicate the results reported in [2].

The code is provided as is and is for academic purpose only. 

Contact voanna AT vision.ee.ethz.ch for questions

# Dependencies
## Python packages
We use Python 3 and Tensorflow 0.10.
The file requirements.txt lists the needed python packages.  To create an Anaconda environment from these, run ``` conda create -n ecc_env --file requirements.txt ```
## Datasets
We use the MNIST, Omniglot and notMNIST datasets for flankers, as well as the MIT places dataset for the backgrounds.  To just run experiments using MNIST, these are not necessary.

* **notMNIST** run ```git clone https://github.com/davidflanagan/notMNIST-to-MNIST.git``` inside the third-party directory.  From the notMNIST-to-MNIST directory, follow instructions 2 & 3 of their README.md
* **Omniglot** run ```git clone https://github.com/brendenlake/omniglot.git``` inside the third-party directory.  From the omniglot/python directory, run `../../src/bash/preprocess_omniglot.sh`. Requires imagemagick.
* **MIT Places** register at http://places.csail.mit.edu/downloadData.html, and then also unzip the dataset inside the `third-party` directory.

# The Model
We can train the eccentricity model from the command line. We can do this by specifying parameters both for the training images and for the model architecture using command line flags

Parameters:
```
--parity
    if "none", uses all MNIST digits, else uses "even" or "odd"
--random_shifts
    generates images with randomly shifted digits (overrides --train_ecc option)
--learning_rate   
    learning rate used in the Adagrad optimizer
--pm
    pooling scheme across scales.  Each number specifies the number of scales remaining at each layer. The first number has to be the same as used in --num_scales. 
--num_scales
    number of scales in input data, which is the same as number of crops
--total_pool
    if true, pool all feature maps to 1x1 size in final layer
--contrast_norm
    contrast normalisation type to use, can be areafactor or None (see [1])
--train_dir
    directory where to write event logs and checkpoint.
--model_name
    model weights are stored inside the directory train_dir/trained/model_name
```

To train a model to recognize only even digits shifted randomly across the visual field, using a learning rate of 0.01 in the Adagrad optimizer, using an 11-1-1-1-1 pooling scheme across scales and total pooling across space, using images without contrast normalisation, we would run:
```
src/python/ecc/ecc_train.py --parity=even --random_shift --learning_rate=0.01 --pm=11-1-1-1-1 --num_scales=11 --total_pool=True --train_dir=experiments/demo-experiment/gen  --contrast_norm=None  --model_name=11-1-1-1-1_total_pool_contrast_None_lr0.01
```

This command would first generate the training images and then train the model.

All commands are meant to be run from the root of this repository. More parameters are available, and can be found at the top of the files `src/python/ecc/ecc.py`, `src/python/ecc/convert_to_records.py`, `src/python/ecc/ecc_train.py`, and `src/python/ecc/ecc_eval.py`, where the flags are defined.
The code organisation is taken from on the tutorial for the CIFAR-10 model in Tensorflow (http://tensorflow.org/tutorials/deep_cnn/)

### Training and testing images
If there does not yet exist a .tfrecords file containing the images to be used for training this network, `ecc_train.py` calls `convert_to_records.py`, which creates the appropriate files. 

Each time a new `*.tfrecord` is created, some images are also written to file for debugging in the experiment inside an `images` directory in the `train_dir` directory specified by the command line flag.
Options relating to images can be found in the FLAGS variable at the top of `convert_to_records.py` Examples of different options are 
* Flanker dataset
* Contrast normalisation
* Number of scales
* Digit size
...

NOTE: to repeat experiments with notMNIST and Omniglot flankers, we set the digit size of the MNIST targets to 120, but the size of the notMNIST or Omniglot flankers to 85.  This is because the MNIST digits themselves are 20x20 pixels, but are padded to 28 x 28, making them effectively 85 pixels high (120 x 20 / 28). 

#### Chevron sampling
[2] and [3] have also studied a variation of the eccentricity model in which larger receptive field sizes do not cover the entire visual field, which [2] refers to as *chevron sampling*. From [2]: 
> Let *c* be the chevron parameter, which characterizes the number of scale channels that are active at any given eccentricity. We constrain the network that at most *c* scale channels are active at any given eccentricity. In practice, for a given crop, this simply means zeroing out all data from the crop *c* layers below, effectively replacing the signal from that region with the black background.

This is also implemented in our model. To set *c*, set the `--chevron` flag to *c*. The default value is `float('inf')` or infinite, as in the experiments described in our paper([1]), all possible channels are active for a given eccentricity. [2] uses a chevron value of 2.

### Model architecture
The code for the model is in src/ecc
Any of the FLAGS (tf.app.flags...) defined in `ecc_eval.py`, `ecc_train.py`, `convert_to_records.py` and `ecc.py` are parameters of both the model and the data that can be changed using the command line flags.

How the model was used in practice to run experiments is described below.  

# An example experiment
A demo of how experiments are run is shown in `experiments/demo-experiment`.  All generated files are placed into a directory called `gen` inside `experiments/demo-experiment`.


For each experiment, all the different models we want to train are specified in `gen_train.py`. This script writes all the models we want to train into files, e.g. `experiments/demo-experiment/gen/scripts/11-1-1-1-1_total_pool_contrast_None_lr0.01.sh`
Here we train each model with three different learning rates.

We evaluate each such model with *a*, *ax*, *xa*, and *xax* flankers at different eccentricities.   We specify the parameters of the testing images in `eval_all.py`

The script called `run_crowding_acc.py` selects the best learning rate for each model, and writes one file to evaluate the each model with a specific combination of target eccentrcity, flanker eccentricity and flanker type (0, 1 or 2 flankers).  These combinations are just referred to by indices in the files, and these indices are used by the `eval_all.py` program.

Since these experiments are easily parellelizable, these scripts have been written to be run on a computing cluster, SLURM in this case.  The instructions below assume use of SLURM.

To run this experiment
0) Make sure the python scripts are on the PYTHONPATH. If not, add them by running `export PYTHONPATH=$PYTHONPATH:/path/to/eccentricity/src/python:/path/to/eccentricity/src/python/ecc`
1) run ```python experiments/demo-experiment/gen_train.py```. This creates a shell script for training each model.  After running this command, the `experiments/demo-experiment/gen/scripts` directory  should contain the following
```
train_11-1-1-1-1_total_pool_contrast_areafactor_lr0.001.sh
train_11-1-1-1-1_total_pool_contrast_areafactor_lr0.01.sh
train_11-1-1-1-1_total_pool_contrast_areafactor_lr0.1.sh
train_11-1-1-1-1_total_pool_contrast_None_lr0.001.sh
train_11-1-1-1-1_total_pool_contrast_None_lr0.01.sh
train_11-1-1-1-1_total_pool_contrast_None_lr0.1.sh
train_all.sh
```

2) submit the jobs to the cluster and wait.  In our case, it is best to open an instance of `screen` and run `./experiments/demo-experiment/scripts/train_all.sh`.
To make sure that two jobs are not writing the same `*.tfrecords` file (data format for feeding images to network) simultaneously and corrupting it, the `train_all.sh` script `sleep`s until all `*.tfrecords` have been written to file and only then submits the rest of the jobs.  This is why we submit the jobs in `screen`.
Logs are saved to `experiments/demo-experiment/gen/trainlogs/`. Trained models are in `experiments/demo-experiment/gen/trained/`

3) When the jobs are done run ```python experiments/demo-experiment/run_crowding_acc.py```. This script picks the model with the best learning rate according to the training accuracy at after 20 epochs, and creates evaluation jobs in the `./experiments/demo/experiment/scripts/` directory.  

4) submit the jobs using `./experiments/demo-experiment/gen/scripts/eval_all.sh` to the cluster in a `screen` and wait for this to complete.
Again, we have included `sleep` commands to make sure that jobs are not creating the same datasets twice.
The jobs write the results for each datapoint into the `/experiments/demo-experiment/gen/precision_{modelname}` directory.  Logs for evaluation are in `/experiments/demo-experiment/gen/evallog_{modelname}`

5) View results in the `summary.ipynb` notebook.  Launch it by running ```jupyter notebook summary.ipynb```

The demo experiment is a template for running similar experiments. To create a new experiment with different settings, make sure that the settings for the experiment match in all three files (`gen_train.py`, `run_crowding_acc.py` and `eval_all.py`) in the places that have been marked with a `# SETTING` comment.

# References
If you use the code, partly or as is, in your projects and papers, please cite:

[1] Volokitin, A., Roig, G., and Poggio, T. Do Deep Neural Networks Suffer from Crowding?. CBMM memo noXX (2017).

[2] Chen, F., Roig, G., Isik, L., Boix, X. and Poggio, T. Eccentricity Dependent Deep Neural Networks: Modeling Invariance in Human Vision. AAAI Spring Symposium Series, Science of Intelligence (2017).

[3] Poggio, T., Mutch, J., Isik, L. Computational role of eccentricity dependent cortical magnification. arXiv:1406.1770, CBMM memo no17, (2014).
