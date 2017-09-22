# -*- coding: utf-8 -*-

""" 
    Test sample list generate
    2017.9.19
"""

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf

tf.app.flags.DEFINE_string('load_path','../data/MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TRAIN/17_DEG/BMP2/SN_9563/',"file directory")
tf.app.flags.DEFINE_integer('target', 0, "BMP2: 0, BTR70: 1, T72: 2")
FLAGS = tf.app.flags.FLAGS

def IsSubString(SubStrList,Str):
    
    flag=True
    for substr in SubStrList:
        if not(substr in Str):
            flag=False
    return flag

def GetFileList(TargetType=FLAGS.target, FindPath=FLAGS.load_path, FlagStr='jpeg'):
    
    text = ''
    FileNames=os.listdir(FindPath)
    if (len(FileNames)>0):
        for fn in FileNames:
            if (len(FlagStr)>0):

                temfile1='%d'%TargetType
                if (IsSubString(FlagStr,fn)):
                    text = text + FindPath + fn + ' ' + temfile1 + '\n'
                    
                else:
                    text = text + FindPath + fn + ' ' + temfile1 + '\n'

    return text

all_the_text = ''

Path = '../data/MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BMP2/SN_9563/'
Type = 0
all_the_text = all_the_text + GetFileList(TargetType = Type, FindPath = Path, FlagStr='jpeg')
print(Path + ' Ready!\n')

Path = '../data/MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BMP2/SN_9566/'
Type = 0
all_the_text = all_the_text + GetFileList(TargetType = Type, FindPath = Path, FlagStr='jpeg')
print(Path + ' Ready!\n')

Path = '../data/MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BMP2/SN_C21/'
Type = 0
all_the_text = all_the_text + GetFileList(TargetType = Type, FindPath = Path, FlagStr='jpeg')
print(Path + ' Ready!\n')

Path = '../data/MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BTR70/SN_C71/'
Type = 1
all_the_text = all_the_text + GetFileList(TargetType = Type, FindPath = Path, FlagStr='jpeg')
print(Path + ' Ready!\n')

Path = '../data/MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/T72/SN_132/'
Type = 2
all_the_text = all_the_text + GetFileList(TargetType = Type, FindPath = Path, FlagStr='jpeg')
print(Path + ' Ready!\n')

Path = '../data/MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/T72/SN_812/'
Type = 2
all_the_text = all_the_text + GetFileList(TargetType = Type, FindPath = Path, FlagStr='jpeg')
print(Path + ' Ready!\n')

Path = '../data/MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/T72/SN_S7/'
Type = 2
all_the_text = all_the_text + GetFileList(TargetType = Type, FindPath = Path, FlagStr='jpeg')
print(Path + ' Ready!\n')

#write filename;
file_object = open('testfilelist.txt', 'w+')
file_object.write(all_the_text)
file_object.close( )
print(' Done!')
