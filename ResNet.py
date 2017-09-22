# -*- coding: utf-8 -*-
""" 
This source code is created by Le Ning (lening@sjtu.edu.cn) on Sep 21, 2017.
This is training code based on resnet.
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

# Building 'residual network'
n = 5
net = tflearn.input_data(shape=[None, FLAGS.sample_size, FLAGS.sample_size, 1])
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

# Regression
net = tflearn.fully_connected(net, num_classes, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=30000, staircase=True)
net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy',learning_rate=0.0001)

# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_sar',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=FLAGS.nepoch,validation_set=(testX,testY),
          snapshot_epoch=False, snapshot_step=100,
          show_metric=True, batch_size=FLAGS.batchsize, shuffle=True,
          run_id='model_resnet_sar')

model.save('model_resnet_sar')
