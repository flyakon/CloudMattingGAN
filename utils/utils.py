"""
Cloud Generation
Description: Utils
Author: Zhengxia Zou
Date: Feb., 2019
"""

import numpy as np
import os
import tensorflow as tf
import cv2
import glob
import initializer as init
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

# All parameters used in this file



import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import OrderedDict
import os

def reshape_list(data,shape=None):
    result=[]
    if shape is None:
        for a in data:
            if isinstance(a,(list,tuple)):
                result+=list(a)
            else:
                result.append(a)
    else:
        i = 0
        for s in shape:
            if s == 1:
                result.append(data[i])
            else:
                result.append(data[i:i + s])
            i += s
    return  result

def smooth_L1(x):
    abs_x=tf.abs(x)
    result=tf.where(abs_x<1,tf.square(x)*0.5,abs_x-0.5)
    return result

def tensor_shape(x,rank=4):
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape=x.get_shape().with_rank(rank).as_list()
        dynamic_shape=tf.unstack(tf.shape(x),rank)
        return [s if s is not None else d for s,d in zip(static_shape,dynamic_shape)]

def get_variables_to_restore(
        scope_to_include, suffix_to_exclude):
    """to parse which var to include and which
    var to exclude"""

    vars_to_include = []
    for scope in scope_to_include:
        vars_to_include += slim.get_variables(scope)

    vars_to_exclude = set()
    for scope in suffix_to_exclude:
        vars_to_exclude |= set(
            slim.get_variables_by_suffix(scope))
    return [v for v in vars_to_include if v not in vars_to_exclude]

def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])

def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
    return var_dict

def data_augmentation(batch,img_size):

    batch_size = batch.shape[0]

    # left-right flip
    if np.random.rand(1) > 0.5:
        batch = batch[:, :, ::-1, :]

    # up-down flip
    if np.random.rand(1) > 0.5:
        batch = batch[:, ::-1, :, :]

    # rotate 90
    if np.random.rand(1) > 0.5:
        for id in range(batch_size):
            batch[id, :, :, :] = np.rot90(batch[id, :, :, :], k=1)  # 90

    # rotate 180
    if np.random.rand(1) > 0.5:
        for id in range(batch_size):
            batch[id, :, :, :] = np.rot90(batch[id, :, :, :], k=2)  # 180

    # rotate 270
    if np.random.rand(1) > 0.5:
        for id in range(batch_size):
            batch[id, :, :, :] = np.rot90(batch[id, :, :, :], k=-1)  # 270

    # random crop and resize 0.5~1.0
    if np.random.rand(1) > 0.5:

        IMG_SIZE = batch.shape[1]
        scale = np.random.rand(1) * 0.5 + 0.5
        crop_height = int(scale * img_size)
        crop_width = int(scale * img_size)
        x_st = int((1 - scale) * np.random.rand(1) * (img_size - 1))
        y_st = int((1 - scale) * np.random.rand(1) * (img_size - 1))
        x_nd = x_st + crop_width
        y_nd = y_st + crop_height

        for id in range(batch_size):
            cropped_img = batch[id, y_st:y_nd, x_st:x_nd, :]
            batch[id, :, :, :] = cv2.resize(cropped_img, dsize=(img_size, img_size))

    return batch




def get_batch(DATA, batch_size, mode,img_size, with_data_augmentation=True):

    if mode is 'cloudimage':
        if np.random.rand(1) > 0.5:
            data = DATA['thick_cloud_images']
        else:
            data = DATA['thin_cloud_images']

    if mode is 'background':
        data = DATA['background_images']

    n, h, w, c = data.shape
    idx = np.random.choice(range(n), batch_size, replace=False)
    batch = data[idx, :, :, :]

    if with_data_augmentation is True:
        batch = data_augmentation(batch,img_size)

    # plt.imshow(batch[0,:,:,:])
    # plt.show()
    # plt.pause(0.5)

    return batch




def plot4x4(samples):

    IMG_SIZE = samples.shape[1]

    img_grid = np.zeros((4 * IMG_SIZE, 4 * IMG_SIZE, 3))

    for i in range(16):
        py, px = IMG_SIZE * int(i / 4), IMG_SIZE * (i % 4)
        this_img = samples[i, :, :, :]
        img_grid[py:py + IMG_SIZE, px:px + IMG_SIZE, :] = this_img

    return img_grid




def plot2x2(samples):

    IMG_SIZE = samples.shape[1]

    img_grid = np.zeros((2 * IMG_SIZE, 2 * IMG_SIZE, 3),np.uint8)

    for i in range(len(samples)):
        py, px = IMG_SIZE * int(i / 2), IMG_SIZE * (i % 2)
        this_img = samples[i, :, :, :]
        this_img=np.uint8(this_img*255)
        img_grid[py:py + IMG_SIZE, px:px + IMG_SIZE, :] = this_img

    return img_grid





def load_historical_model(sess, checkpoint_dir='checkpoints'):

    # check and create model dir
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)

    if 'checkpoint' in os.listdir(checkpoint_dir):
        # training from the last checkpoint
        print('loading model from the last checkpoint ...')
        saver = tf.train.Saver(get_variables_to_restore(['Generator_scope','Discriminator_scope'],
                                                        ['Adam','Adam_1']))
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, latest_checkpoint)
        print(latest_checkpoint)
        print('loading finished!')
    else:
        print('no historical model found, start training from scratch!')




def load_images(image_dir,img_size):

    data = {
        'background_images': 0,
        'thick_cloud_images': 0,
        'thin_cloud_images': 0,
    }

    # load background images
    img_dirs = glob.glob(os.path.join(image_dir, 'bg_slice/*.jpg'))
    m_tr_imgs = len(img_dirs)
    image_buff = np.zeros((m_tr_imgs, img_size, img_size, 3))

    for i in range(m_tr_imgs):
        file_name = img_dirs[i]
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size)) / 255.
        image_buff[i, :, :, :] = img

        i += 1
        if np.mod(i, 20) == 0:
            print('reading background images: ' + str(i) + ' / ' + str(m_tr_imgs))
    data['background_images'] = image_buff


    # load thick cloud images
    img_dirs = glob.glob(os.path.join(image_dir, 'thick_slice/*.jpg'))
    m_tr_imgs = len(img_dirs)
    image_buff = np.zeros((m_tr_imgs, img_size, img_size, 3))

    for i in range(m_tr_imgs):
        file_name = img_dirs[i]
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size)) / 255.
        image_buff[i, :, :, :] = img
        i += 1
        if np.mod(i, 20) == 0:
            print('reading thick cloud images: ' + str(i) + ' / ' + str(m_tr_imgs))
    data['thick_cloud_images'] = image_buff



    # load thin cloud images and masks
    img_dirs = glob.glob(os.path.join(image_dir, 'thin_slice/*.jpg'))
    m_tr_imgs = len(img_dirs)
    image_buff = np.zeros((m_tr_imgs, img_size, img_size, 3))

    for i in range(m_tr_imgs):
        file_name = img_dirs[i]
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size)) / 255.
        image_buff[i, :, :, :] = img
        i += 1
        if np.mod(i, 20) == 0:
            print('reading thin cloud images: ' + str(i) + ' / ' + str(m_tr_imgs))
    data['thin_cloud_images'] = image_buff

    print('done done done')

    return data



