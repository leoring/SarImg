# -*- coding: utf-8 -*-
""" 
This source code is created by Le Ning (lening@sjtu.edu.cn) on Sep 19, 2017.
This is training code based on Vgg Net.
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf														
import tflearn
import numpy as np

from tflearn.data_utils import shuffle, to_categorical
import tflearn.data_utils as du

#define paramaters
tf.app.flags.DEFINE_integer('sample_size', 32, "sample size")
tf.app.flags.DEFINE_integer('nepoch', 1000, "epoch number")
tf.app.flags.DEFINE_integer('batchsize', 32, "batch size number")
FLAGS = tf.app.flags.FLAGS

# Data loading
# Load path/class_id image file:
train_file = 'trainfilelist.txt'
test_file = 'testfilelist.txt'

# Build the preloader array, resize images to sample size
from tflearn.data_utils import image_preloader
X, Y = image_preloader(train_file, image_shape=(FLAGS.sample_size, FLAGS.sample_size),
                       mode='file', categorical_labels=True,
                       normalize=True)

testX, testY = image_preloader(test_file, image_shape=(FLAGS.sample_size, FLAGS.sample_size),
                       mode='file', categorical_labels=True,
                       normalize=True)

#Reshape X and testX
X = np.array(X)
X = X.reshape([-1, FLAGS.sample_size, FLAGS.sample_size, 1])

testX = np.array(testX)
testX = testX.reshape([-1, FLAGS.sample_size, FLAGS.sample_size, 1])

X, mean = du.featurewise_zero_center(X)
testX = du.featurewise_zero_center(testX, mean)

num_classes = 3

# Building 'vgg network'
network = tflearn.input_data(shape=[None, FLAGS.sample_size, FLAGS.sample_size, 1])
network = tflearn.conv_2d(network, 64, 3, activation='relu')
network = tflearn.conv_2d(network, 64, 3, activation='relu')
network = tflearn.max_pool_2d(network, 2, strides=2)

network = tflearn.conv_2d(network, 128, 3, activation='relu')
network = tflearn.conv_2d(network, 128, 3, activation='relu')
network = tflearn.max_pool_2d(network, 2, strides=2)

network = tflearn.conv_2d(network, 256, 3, activation='relu')
network = tflearn.conv_2d(network, 256, 3, activation='relu')
network = tflearn.conv_2d(network, 256, 3, activation='relu')
network = tflearn.max_pool_2d(network, 2, strides=2)

network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.max_pool_2d(network, 2, strides=2)

network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.conv_2d(network, 512, 3, activation='relu')
network = tflearn.max_pool_2d(network, 2, strides=2)

network = tflearn.fully_connected(network, 4096, activation='relu')
network = tflearn.dropout(network, 0.5)
network = tflearn.fully_connected(network, 4096, activation='relu')
network = tflearn.dropout(network, 0.5)
network = tflearn.fully_connected(network, num_classes, activation='softmax')

network = tflearn.regression(network, optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=FLAGS.nepoch, validation_set=(testX,testY), shuffle=True,
          show_metric=True, batch_size=FLAGS.batchsize, snapshot_step=100,
          snapshot_epoch=False, run_id='vgg_module_sar')

model.save('vgg_module_sar')
