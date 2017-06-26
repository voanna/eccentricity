"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np
np.random.seed(seed=2017)
import multiscale
import time
import shutil
from PIL import Image
import functools
import glob

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'  # MNIST filenames
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

#These variables are only used to create the name of the tfrecord files. Note that not all possible parameters are included.
FLAGS_FOR_NAME = []
ABBR_FOR_NAME = {}

tf.app.flags.DEFINE_integer('validation_size', 5000,
                            'Number of examples to separate from the training '
                            'data for the validation set.')

tf.app.flags.DEFINE_string('train_dir', 'experiments/experiment-name/gen/',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('model_name', 'model_name', 'model weights are stored inside the directory train_dir/trained/model_name')

tf.app.flags.DEFINE_string('fovea', 'exponential', 'Defines whether multiscale crops are linearly or exponentially spaced')

tf.app.flags.DEFINE_integer('crop_size', 60, 'Size of crops in pixels')

tf.app.flags.DEFINE_integer('field_size', 1920, 'Non-downsampled image size in which to embed mnist image')

FLAGS_FOR_NAME.append('digit_height')
ABBR_FOR_NAME['digit_height'] = 'dh'
tf.app.flags.DEFINE_integer('digit_height', 120, 'Size of digit to embed in non-resized image')

FLAGS_FOR_NAME.append('flanker_height')
ABBR_FOR_NAME['flanker_height'] = 'flanker_h'
tf.app.flags.DEFINE_integer('flanker_height', 120, 'Size of flanker to embed in non-resized image')

FLAGS_FOR_NAME.append('train_ecc')
ABBR_FOR_NAME['train_ecc'] = 'train_ecc'
tf.app.flags.DEFINE_float('train_ecc', 0, 'Eccentricity at which to place digits')

FLAGS_FOR_NAME.append('num_scales')
ABBR_FOR_NAME['num_scales'] = 'ns'
tf.app.flags.DEFINE_integer('num_scales', 6, 'number of scales in input data, which is the same as number of crops')

FLAGS_FOR_NAME.append('random_shifts')
ABBR_FOR_NAME['random_shifts'] = 'rs'
tf.app.flags.DEFINE_boolean('random_shifts', False, 'generates images with randomly shifted digits (overrides --train_ecc option)')

FLAGS_FOR_NAME.append('parity')
ABBR_FOR_NAME['parity'] = 'p'
tf.app.flags.DEFINE_string('parity', "none", 'If "none", uses all MNIST digits, else uses "even" or "odd"')

FLAGS_FOR_NAME.append('flanker_type')
ABBR_FOR_NAME['flanker_type'] = 'ftype'
tf.app.flags.DEFINE_integer('flanker_type', 0, '0 - no flankers, 1 - one, 2 - two symmetrical')

FLAGS_FOR_NAME.append('target_ecc')
ABBR_FOR_NAME['target_ecc'] = 'te'
tf.app.flags.DEFINE_float('target_ecc', 0, 'Eccentricity of target center')

FLAGS_FOR_NAME.append('flanker_ecc')
ABBR_FOR_NAME['flanker_ecc'] = 'fe'
tf.app.flags.DEFINE_float('flanker_ecc', 0, 'Eccentricity of flanker center')

FLAGS_FOR_NAME.append('chevron')
ABBR_FOR_NAME['chevron'] = 'c'
tf.app.flags.DEFINE_float('chevron', float('inf'), 'Eccentricity of flanker center')

FLAGS_FOR_NAME.append('flanker_dset')
ABBR_FOR_NAME['flanker_dset'] = 'fd'
tf.app.flags.DEFINE_string('flanker_dset', 'MNIST', 'Use MNIST, notMNIST or omniglot for flanker objects')
FLAGS = tf.app.flags.FLAGS

FLAGS_FOR_NAME.append('contrast_norm')
ABBR_FOR_NAME['contrast_norm'] = 'cnorm'
tf.app.flags.DEFINE_string('contrast_norm', 'None', 'Contrast normalisation type to use, can be areafactor or None')

FLAGS_FOR_NAME.append('place')
ABBR_FOR_NAME['place'] = 'place'
tf.app.flags.DEFINE_string('place', 'None', 'Background from places to use.  None means black background, Random means random from any category in background dict (see src)')

tf.app.flags.DEFINE_integer('image_save_interval', '1000', 'How often to save the generated images to file')

tf.app.flags.DEFINE_string('places_dir', 'third-party/places/data/vision/torralba/deeplearning/images256/',
                           "Directory where to places dataset is ")
FLAGS = tf.app.flags.FLAGS

background_dict = {
  0: 'a/abbey',
  1: 'a/airport_terminal',
  2: 'a/alley',
  3: 'a/amusement_park',
  4: 'a/aquarium',
  5: 'a/arch',
  6: 'a/art_gallery',
  7: 'a/auditorium',
  8: 'b/badlands',
  9: 'b/ballroom'

}

def filelist(category):
  imdir = os.path.join(FLAGS.places_dir, category)
  images = sorted(glob.glob(imdir + '/*.jpg'))
  assert len(images)==15000, "Using category {} with {} images, expected 15000".format(category, len(images))
  return images

def background(category, name, index):
  images = filelist(category)
  if name == 'train':
    offset = 0
  elif name == 'validation':
    offset = 7000
  elif name == 'test':
    offset = 12000
  else:
    raise ValueError("Dataset type is not train, validation or test. Is {}".format(name))

  im = Image.open(images[index + offset]).convert('L')
  return im

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(data_set, name, flanker_dset='MNIST'):
  images = data_set.images
  labels = data_set.labels

  assert FLAGS.parity == 'even', ("This implementation" 
    " assumes targets are even and flankers are odd."
    " There are fewer even examples, so swapping will"
    " break this code")
  even_mask = labels % 2 == 0
  odd_mask = labels % 2 == 1

  assert sum(even_mask) <= sum(odd_mask), ("Number of even "
    "should be less than odd, got {} even and {} odd ".format(
      sum(even_mask), sum(odd_mask)))

  print('Num targets = {}'.format(len(even_mask)))

  targets = images[even_mask]
  if flanker_dset == 'MNIST':
    flankers = images[odd_mask]
  elif flanker_dset == 'notMNIST':
    data_sets = mnist.read_data_sets('third-party/notMNIST-to-MNIST/',
                                     dtype=tf.uint8,
                                     reshape=False)
    if name == 'train':
      flankers = data_sets.train.images
    elif name == 'validation':
      flankers = data_sets.validation.images
    else:
      flankers = data_sets.test.images
  elif flanker_dset == 'omniglot':
    data_sets = mnist.read_data_sets('third-party/omniglot/python/',
                                     dtype=tf.uint8,
                                     reshape=False)
    if name == 'train':
      flankers = data_sets.train.images
    elif name == 'validation':
      flankers = data_sets.validation.images
    else:
      flankers = data_sets.test.images
  else:
    raise ValueError('flanker_dset can be MNIST, notMNIST, or omniglot, got {}'.format(flanker_dset))

  remap = {2*i:i for i in range(5)}

  labels = [remap[label] for label in labels[even_mask]]

  num_examples = sum(even_mask)

  if targets.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (targets.shape[0], num_examples))
  rows = FLAGS.crop_size
  cols = FLAGS.crop_size
  scales = FLAGS.num_scales
  depth = 1
  
  filename = os.path.join(FLAGS.train_dir, 'data', name + '_' + tfrecord_name() + '.tfrecords')
 
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename + '.partial')
  start_time = time.time()

  imgdir = os.path.join(
    FLAGS.train_dir, 'images', tfrecord_name(), name)

  if tf.gfile.Exists(imgdir):
    tf.gfile.DeleteRecursively(imgdir)
  tf.gfile.MakeDirs(imgdir)

 
  for index in range(num_examples):  
    if FLAGS.place == 'random':
      counter = {background_dict[i]:0 for i in range(10)}
      # counter variable is there to keep track of the next example of background. 
      category = background_dict[np.random.randint(10)]
      counter[category] += 1
      bground = background(category, name, counter[category])

    elif FLAGS.place == 'None':
      category = None
      bground = None

    else:
      raise ValueError('Type of background with places can be "random" or "None".  '
        'Got {}'.format(FLAGS.place))

    if index % FLAGS.image_save_interval == 0:
      duration = time.time() - start_time
      start_time = time.time()
      print('{} of {}, {} sec since last print'.format(
        index, num_examples, duration))

    if FLAGS.random_shifts:
      width = FLAGS.field_size - FLAGS.digit_height
      if FLAGS.flanker_type == 0:
        target_ecc = np.random.random_sample() * width - width/float(2)
        flanker_ecc = FLAGS.flanker_ecc
      elif FLAGS.flanker_type == 1:
        spacing = FLAGS.flanker_ecc - FLAGS.target_ecc
        #rejection sampling 
        sample_valid = False
        while not sample_valid:
          target_ecc = np.random.random_sample() * width - width/float(2)
          flanker_ecc = target_ecc + spacing
          min_left_ecc = min(target_ecc - FLAGS.digit_height/float(2.0), flanker_ecc - FLAGS.flanker_height/float(2.0))
          max_right_ecc = max(target_ecc + FLAGS.digit_height/float(2.0), flanker_ecc + FLAGS.flanker_height/float(2.0))
          if max_right_ecc < FLAGS.field_size / float(2.0) and min_left_ecc > -FLAGS.field_size / float(2):
            sample_valid = True
      else:
        spacing = FLAGS.flanker_ecc - FLAGS.target_ecc
        sample_valid = False
        while not sample_valid:
          target_ecc = np.random.random_sample() * width - width/float(2)
          flanker_ecc = target_ecc + spacing
          opp_ecc = 2*target_ecc - flanker_ecc
          min_left_ecc  = min(opp_ecc - FLAGS.flanker_height/float(2.0), flanker_ecc - FLAGS.flanker_height/float(2.0))
          max_right_ecc = max(opp_ecc + FLAGS.flanker_height/float(2.0), flanker_ecc + FLAGS.flanker_height/float(2.0))
          if max_right_ecc < FLAGS.field_size / float(2.0) and min_left_ecc > -FLAGS.field_size / float(2.0):
            sample_valid = True

    else:
      target_ecc = FLAGS.target_ecc
      flanker_ecc = FLAGS.flanker_ecc

    if index % FLAGS.image_save_interval == 0:
      ms = multiscale.build_multiscale(
        targets[index], 
        FLAGS.crop_size,         
        FLAGS.field_size, 
        FLAGS.digit_height,
        FLAGS.num_scales, 
        target_ecc, 
        fovea=FLAGS.fovea, 
        flanker_type=FLAGS.flanker_type,
        flanker=flankers[index],
        flanker_ecc=flanker_ecc,
        flanker_height=FLAGS.flanker_height,
        chevron=FLAGS.chevron,
        contrast_norm=FLAGS.contrast_norm,
        background=bground,
        category=category,
        save=True, 
        save_directory=imgdir, 
        save_basename=str(index))

    ms = multiscale.build_multiscale(
      targets[index], 
      FLAGS.crop_size,         
      FLAGS.field_size, 
      FLAGS.digit_height,
      FLAGS.num_scales, 
      target_ecc, 
      fovea=FLAGS.fovea, 
      chevron=FLAGS.chevron,
      contrast_norm=FLAGS.contrast_norm,
      background=bground,
      category=category,
      flanker_type=FLAGS.flanker_type,
      flanker=flankers[index],
      flanker_ecc=flanker_ecc,
      flanker_height=FLAGS.flanker_height,)


    image_raw = ms.tostring()
    example = tf.train.Example(
      features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'scales': _int64_feature(scales),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()
  os.rename(filename + '.partial', filename)


def tfrecord_name():
  _ = FLAGS.crop_size
  dictionary = FLAGS.__flags

  keys = FLAGS_FOR_NAME
  parts = ['{}={}'.format(ABBR_FOR_NAME[k], dictionary[k]) for k in keys]
  name = '-'.join(parts)
  return name

def preprocess_training():
  data_sets = mnist.read_data_sets(os.path.join(FLAGS.train_dir, 'data'),
                                   dtype=tf.uint8,
                                   reshape=False)

  # Convert to Examples and write the result to TFRecords.
  convert_to_tfrecord(data_sets.train, 'train', flanker_dset=FLAGS.flanker_dset)
  convert_to_tfrecord(data_sets.validation, 'validation', flanker_dset=FLAGS.flanker_dset)
  convert_to_tfrecord(data_sets.test, 'test', flanker_dset=FLAGS.flanker_dset)

def maybe_download_and_preprocess_training():
  print('Settings used are:')
  f = FLAGS.__dict__['__flags']
  for key in sorted(f.keys()):
    print('{} : {}, type {}'.format(key, f[key], type(f[key])))

  filename = os.path.join(FLAGS.train_dir, 
    'data', 'train_' + tfrecord_name() + '.tfrecords')
  print(filename)
  if not os.path.exists(filename):
    preprocess_training()

def preprocess_evaluation():
  data_sets = mnist.read_data_sets(os.path.join(FLAGS.train_dir, 'data'),
                                   dtype=tf.uint8,
                                   reshape=False)

  convert_to_tfrecord(data_sets.test, 'test', flanker_dset=FLAGS.flanker_dset)

def maybe_download_and_preprocess_evaluation():
  print('Settings used are:')
  f = FLAGS.__dict__['__flags']
  for key in sorted(f.keys()):
    print('{} : {}, type {}'.format(key, f[key], type(f[key])))


  filename = os.path.join(FLAGS.train_dir, 
    'data', 'test_' + tfrecord_name() + '.tfrecords')
  print(filename)
  try:
    if not os.path.exists(filename):
      preprocess_evaluation()
  except multiscale.InvalidFlankerPosition:
    os.remove(filename + '.partial')
    imgdir = os.path.join(
      FLAGS.train_dir, 'images', tfrecord_name(), 'test')
    shutil.rmtree(imgdir) 
    raise
def main(argv):
  pass

if __name__ == '__main__':
  tf.app.run()
