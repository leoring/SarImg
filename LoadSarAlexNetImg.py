# -*- coding: utf-8 -*-
""" 
This source code is created by Le Ning (lening@sjtu.edu.cn) on September 20, 2017.
This is image prediction code based on alexnet.
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf														
import tflearn
from PIL import Image
import numpy as np
import os
import tflearn.data_utils as du

#define paramaters
tf.app.flags.DEFINE_integer('sample_size', 128, "sample size")
tf.app.flags.DEFINE_string('targetfile','HB04973.004.jpeg',"Target picture")
tf.app.flags.DEFINE_string('inputpath','../data/MSTAR-PublicT72Variants-CD1/MSTAR_PUBLIC_T_72_VARIANTS_CD1/15_DEG/COL2/SCENE1/A64/',"Target picture directory")
tf.app.flags.DEFINE_integer('modeltype', 0, "AlexNet: 0, Vgg: 1")
tf.app.flags.DEFINE_string('alexmodelfile','alex_net_sar',"alex model file")
tf.app.flags.DEFINE_string('vggmodelfile','vgg_module_sar',"vgg model file")
tf.app.flags.DEFINE_boolean('filemode', True, "file:True or directory:False")
FLAGS = tf.app.flags.FLAGS

#Directly read data from images;
def DirectReadImg(filename, Imgheight=FLAGS.sample_size, Imgwidth=FLAGS.sample_size, normlized=True):

    #Load image from disk
    img=np.array(Image.open(filename).resize((Imgheight,Imgwidth)))
    
    if(normlized == True):
        img_normed = img/255.0
        return img_normed.reshape(1,Imgheight,Imgwidth,1)
    else:
        return img.reshape(1,Imgheight,Imgwidth,1)

def create_vgg(num_classes = 3, imgSize = 32):

    # Building 'vgg network'
    network = tflearn.input_data(shape=[None, imgSize, imgSize, 1])
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
    return network

def create_alexnet(num_classes = 3, imgSize = FLAGS.sample_size):

    # Building 'AlexNet'
    network = tflearn.input_data(shape=[None, imgSize, imgSize, 1])
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
                         learning_rate=0.001)
    return network

def IsSubString(SubStrList,Str):
    
    flag=True
    for substr in SubStrList:
        if not(substr in Str):
            flag=False
    return flag

def RecognizeImg(model, mean, FileName = FLAGS.targetfile):

    imgsize = FLAGS.sample_size
    if(FLAGS.modeltype == 1):
        imgsize = 32    

    X = DirectReadImg(FileName, Imgheight=imgsize, Imgwidth=imgsize, normlized=True)
    X = du.featurewise_zero_center(X, mean)

    pred = model.predict([X[0]])
    print(pred)

    return pred

def DirectoryFileEnvaluation(model, mean, Findpath = FLAGS.inputpath, FlagStr=[]):
    
    FileNames = os.listdir(Findpath)
   
    resList = []
    if (len(FileNames)>0):
        FileNames.sort()
        for fn in FileNames:
            if (len(FlagStr)>0):
                if (IsSubString(FlagStr,fn)):
                   
                    res = RecognizeImg(model, mean, Findpath + fn)  
                    resList.append(res)
                else:
                    
                    res = RecognizeImg(model, mean, Findpath + fn)  
                    resList.append(res)

    return resList

if __name__ == '__main__':
    
    # Data loading
    # Load path/class_id image file:
    train_file = 'trainfilelist.txt'

    imgsize = FLAGS.sample_size
    if(FLAGS.modeltype == 1):
        imgsize = 32

    # Build the preloader array, resize images to sample size
    from tflearn.data_utils import image_preloader
    X, Y = image_preloader(train_file, image_shape=(imgsize, imgsize),
                           mode='file', categorical_labels=True,
                           normalize=True)

    #Reshape X and testX
    X = np.array(X)
    X = X.reshape([-1, imgsize, imgsize, 1])

    X, mean = du.featurewise_zero_center(X)

    #load alexnet model
    network = create_alexnet(num_classes = 3, imgSize = 128)
    model = tflearn.DNN(network)
    model.load(FLAGS.alexmodelfile) 

    if(FLAGS.modeltype == 1):
        #load vgg model
        network = create_vgg(num_classes = 3, imgSize = 32)
        model = tflearn.DNN(network)
        model.load(FLAGS.vggmodelfile) 
    
    if(FLAGS.filemode):
        RecognizeImg(model, mean, FLAGS.targetfile)
    else:
        DirectoryFileEnvaluation(model, mean, FLAGS.inputpath, 'jpeg')        

