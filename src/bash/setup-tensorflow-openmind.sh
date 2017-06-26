#!/bin/sh
# Instructions on setting up tensorflow in openmind.
# makes virtual environment in the /om/user/username/tf directory
# To keep the environment variables set by the script and 
# to be able to use tensorflow directly after executing,
# run 'source setup-tensorflow-openmind.sh'
# rather than './setup-tensorflow-openmind.sh'
# For more details, refer to 
# https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#download-and-setup

echo "Clearing all loaded modules, so no conflict of cuda versions..."
echo 'y' | module clear
module load openmind/cudnn/7.5-5.1
module load openmind/cuda/7.5
echo "Clearing pythonpath to ignore old numpy and scipy versions..."
export PYTHONPATH=

echo "Creating a virtual envirnonment for tensorflow with python 3.4..."
echo 'y' | conda create -p /om/user/$(whoami)/tf python=3.4
export PATH=$PATH:/cm/shared/openmind/anaconda/2.1.0/bin/conda
source activate /om/user/$(whoami)/tf

echo "Installing tensorflow 0.10..."
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp34-cp34m-linux_x86_64.whl
echo 'y' | conda install pip
pip install --upgrade --ignore-installed $TF_BINARY_URL

echo ' '
echo ' '
echo ' '
echo 'Installation completed! To check whether it works execute the following:'
echo ' '
echo python -m tensorflow.models.image.mnist.convolutional
echo ' '
echo 'Note that this will create a data/ directory within the current working directory, which you can delete afterwards'
