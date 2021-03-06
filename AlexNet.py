# -*- coding: utf-8 -*-
""" 
This source code is created by Le Ning (lening@sjtu.edu.cn) on Sep 19, 2017.
This is training code based on alexnet.
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf														
import tflearn
import numpy as np

from tflearn.data_utils import shuffle, to_categorical
import tflearn.data_utils as du

#define paramaters
tf.app.flags.DEFINE_integer('sample_size', 128, "sample size")
tf.app.flags.DEFINE_integer('nepoch', 1000, "epoch number")
tf.app.flags.DEFINE_integer('batchsize', 128, "batch size number")
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

# Building 'alexnet network'
network = tflearn.input_data(shape=[None, FLAGS.sample_size, FLAGS.sample_size, 1])
network = tflearn.conv_2d(network, 96, 11, strides=4, activation='relu')
network = tflearn.max_pool_2d(network, 3, strides=2)
network = tflearn.local_response_normalization(network)
network = tflearn.conv_2d(network, 256, 5, activation='relu')
network = tflearn.max_pool_2d(network, 3, strides=2)
network = tflearn.local_response_normalization(network)
network = tflearn.conv_2d(network, 384, 3, activation='relu')
network = tflearn.conv_2d(network, 384, 3, activation='relu')
network = tflearn.conv_2d(network, 256, 3, activation='relu')
network = tflearn.max_pool_2d(network, 3, strides=2)
network = tflearn.local_response_normalization(network)
network = tflearn.fully_connected(network, 4096, activation='tanh')
network = tflearn.dropout(network, 0.5)
network = tflearn.fully_connected(network, 4096, activation='tanh')
network = tflearn.dropout(network, 0.5)
network = tflearn.fully_connected(network, num_classes, activation='softmax')
network = tflearn.regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet_sar',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

#model.load('alex_net_sar')
model.fit(X, Y, n_epoch=FLAGS.nepoch,validation_set=(testX,testY),
          snapshot_epoch=False, snapshot_step=100,
          show_metric=True, batch_size=FLAGS.batchsize, shuffle=True,
          run_id='alexnet_sar')

model.save('alex_net_sar1')
