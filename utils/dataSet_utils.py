
'''
Cloud Mating GAN
Description: Data prepare utils
Author: Wenyuan Li
Date: Feb., 2019
'''
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

import glob
from utils import path_utils


class MattingPrepare(object):

    def get_testDataSet(self,dataSet_path,numsamples=50):
        dataSet_pattern = dataSet_path
        keys_to_features = {
            'img': tf.FixedLenFeature([], tf.string),
            'img/format': tf.FixedLenFeature([], tf.string, default_value='png'),
            'img/height': tf.FixedLenFeature([], tf.int64),
            'img/width': tf.FixedLenFeature([], tf.int64),
            'img/channel': tf.FixedLenFeature([], tf.int64),
            'img/name':tf.FixedLenFeature([],tf.string)
        }
        items_to_handlers = {
            'img': slim.tfexample_decoder.Image('img', 'img/format'),
            'name':slim.tfexample_decoder.Tensor('img/name')
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
        dataSet = slim.dataset.Dataset(data_sources=dataSet_pattern, reader=tf.TFRecordReader,
                                       decoder=decoder, num_samples=numsamples, items_to_descriptions={})
        return dataSet
    def get_dataSet(self,dataSet_path,numsamples=50):
        dataSet_pattern=dataSet_path
        keys_to_features={
            'img':tf.FixedLenFeature([],tf.string),
            'alpha': tf.FixedLenFeature([], tf.string),
            'label':tf.FixedLenFeature([],tf.string),
            'matting': tf.FixedLenFeature([],tf.string),
            'img/format':tf.FixedLenFeature([],tf.string,default_value='png'),
            'img/height': tf.FixedLenFeature([],tf.int64),
            'img/width': tf.FixedLenFeature([],tf.int64),
            'img/channel': tf.FixedLenFeature([],tf.int64),
        }
        items_to_handlers={
            'img':slim.tfexample_decoder.Image('img','img/format'),
            'label': slim.tfexample_decoder.Image('label', 'img/format',channels=1),
            'alpha': slim.tfexample_decoder.Image('alpha', 'img/format',channels=1),
            'matting': slim.tfexample_decoder.Image('matting', 'img/format',channels=1),
        }
        decoder=slim.tfexample_decoder.TFExampleDecoder(keys_to_features,items_to_handlers)
        dataSet=slim.dataset.Dataset(data_sources=dataSet_pattern,reader=tf.TFRecordReader,
                                     decoder=decoder,num_samples=numsamples,items_to_descriptions={})
        return dataSet

    def preprocess(self,image,alpha,matting=None,img_size=512,mode=0):
        """
        对图片进行预处理：
        :param image: 源图像
        :param label: 标签图
        :return: 返回三个尺寸的图像，large尺寸的标签图
        """
        image=tf.image.convert_image_dtype(image,tf.float32)
        image=tf.image.resize_images(image,[img_size,img_size])
        if alpha is not None:
            if mode==1:
                alpha=tf.cast(alpha,tf.float32)/255.
            else:
                alpha = tf.cast(alpha, tf.float32)
            alpha=tf.image.resize_images(alpha,[img_size,img_size])
        if  matting is not None:
            if mode==1:
                matting=tf.cast(matting,tf.float32)/255.
            else:
                matting = tf.cast(matting, tf.float32)
            matting=tf.image.resize_images(matting,[img_size,img_size])
        return image,alpha,matting

    def flip_left_right(self,img,alpha,matting):
        img=tf.image.flip_left_right(img)

        alpha = tf.image.flip_left_right(alpha)
        matting = tf.image.flip_left_right(matting)
        return img,alpha,matting

    def flip_up_dowm(self,img,alpha,matting):
        img=tf.image.flip_up_down(img)

        alpha = tf.image.flip_up_down(alpha)
        matting = tf.image.flip_up_down(matting)
        return img,alpha,matting

    def rot90(self,img,alpha,matting):
        img=tf.image.rot90(img)

        alpha = tf.image.rot90(alpha)
        matting = tf.image.rot90(matting)
        return img,alpha,matting

    def crop_image(self,img,alpha,matting,max_ratios=1,min_ratios=0.8,img_size=512):
        ratios=tf.random_uniform([1],maxval=max_ratios,minval=min_ratios,dtype=tf.float32)
        size = tf.cast(tf.floor(img_size*ratios),tf.int32)[0]
        offset = tf.random_uniform([2], 0, img_size+1-size,dtype=tf.int32)

        img=tf.image.crop_to_bounding_box(img,offset[0],offset[1],size,size)

        alpha = tf.image.crop_to_bounding_box(alpha, offset[0], offset[1], size, size)
        matting = tf.image.crop_to_bounding_box(matting, offset[0], offset[1], size, size)

        img=tf.image.resize_images(img,[img_size,img_size])
        alpha = tf.image.resize_images(alpha, [img_size, img_size])
        matting = tf.image.resize_images(matting, [img_size, img_size])
        return img,alpha,matting

    def original_image(self,img,alpha,matting):

        return img,alpha,matting

    def data_argument(self,img,alpha,matting):
        dst_image = img
        dst_alpha=alpha
        dst_matting=matting

        for _ in range(3):
            index = tf.random_uniform([1], 0, 2, dtype=tf.int32)
            dst_image, dst_alpha, dst_matting = \
                tf.cond(tf.equal(index, 0)[0], lambda: self.rot90(dst_image,dst_alpha,dst_matting),
                             lambda: self.original_image(dst_image,dst_alpha,dst_matting))

        index = tf.random_uniform([1], 0, 2, dtype=tf.int32)
        dst_image, dst_alpha, dst_matting = \
            tf.cond(tf.equal(index,0)[0],lambda :self.crop_image(dst_image,dst_alpha,dst_matting),
                         lambda :self.original_image(dst_image,dst_alpha,dst_matting))

        index = tf.random_uniform([1], 0, 2, dtype=tf.int32)
        dst_image, dst_alpha, dst_matting =\
            tf.cond(tf.equal(index, 0)[0], lambda: self.flip_left_right(dst_image,dst_alpha,dst_matting),
                                       lambda: self.original_image(dst_image,dst_alpha,dst_matting))

        index = tf.random_uniform([1], 0, 2, dtype=tf.int32)
        dst_image, dst_alpha, dst_matting = \
            tf.cond(tf.equal(index, 0)[0], lambda: self.flip_up_dowm(dst_image,dst_alpha,dst_matting),
                                       lambda: self.original_image(dst_image,dst_alpha,dst_matting))

        tf.summary.image('img', tf.expand_dims(img, 0))
        tf.summary.image('argu_img',tf.expand_dims(dst_image,0))
        tf.summary.image('argu_alpha', tf.expand_dims(dst_alpha, 0))
        tf.summary.image('argu_matting', tf.expand_dims(dst_matting, 0))

        return dst_image, dst_alpha, dst_matting

    def split_labels(self,label):
        """
        将标签图(背景为0，云为128，雪为255)进行编码，得到三个标签图：云和背景；雪和背景；云雪和背景
        背景为0；云为1；雪为(1)2
        :param label: 
        :return: 
        """
        label=label[...,0]
        label=tf.cast(label*255,tf.uint8)
        label=tf.cast(label,tf.int64)
        # label=tf.expand_dims(label,axis=-1)
        cloud_mask=tf.logical_and(label>100,label>200)
        cloud_label=tf.where(cloud_mask,tf.ones_like(label),tf.zeros_like(label))
        return cloud_label


    def prepare_labeledData(self,data_path,data_format='*_[0-9].png',
                            reflectance_suffix='_reflectance.png',alpha_suffix='_alpha.png'):
        '''
        获得带标签的图像的路径及标签的路径
        要求图像与标签具有相同的文件名，存放在不同的文件夹中
        :param data_path: 图像文件的保存文件夹,nask_label(label)、reflectance_label和alpha_label在一个文件夹中
        :param data_format:图像的格式，支持通配符
        :return: 返回图像路径和标签图像路径列表
        '''
        data_list=[]
        reflectance_list=[]
        alpha_list=[]

        data_files=glob.glob(os.path.join(data_path,data_format))
        for data_file in data_files:
            file_name= path_utils.get_filename(data_file, is_suffix=False)
            file_name=file_name.replace('_image','')
            reflectance_file = os.path.join(data_path, '%s%s' % (file_name, reflectance_suffix))
            if not os.path.exists(reflectance_file):
                print('%s reflectance file does not exist' % file_name)
                continue

            alpha_file = os.path.join(data_path, '%s%s' % (file_name, alpha_suffix))
            if not os.path.exists(alpha_file):
                print('%s alpha file does not exist' % file_name)
                continue

            data_list.append(data_file)
            reflectance_list.append(reflectance_file)
            alpha_list.append(alpha_file)
        return data_list,alpha_list,reflectance_list

    def prepare_unlabeledData(self,data_path,data_format='*.png'):
        '''
        获得测试图像的路径
        :param data_path: 
        :param data_format: 
        :return: 
        '''
        data_list=[]
        data_files = glob.glob(os.path.join(data_path,data_format))
        for data_file in data_files:
            data_list.append(data_file)
        return data_list


class SegmentPrepare(object):

    def preprocess(self,image, label=None, img_size=512):
        """
		对图片进行预处理：
		:param image: 源图像
		:param label: 标签图
		:return: 返回三个尺寸的图像，large尺寸的标签图
		"""
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [img_size, img_size])
        if label is not None:
            label = tf.cast(label, tf.float32)
            label = tf.image.resize_images(label, [img_size, img_size],tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image,label

    def flip_left_right(self,img, label):
        img = tf.image.flip_left_right(img)
        label = tf.image.flip_left_right(label)

        return img, label

    def flip_up_dowm(self,img, label):
        img = tf.image.flip_up_down(img)

        label = tf.image.flip_up_down(label)

        return img, label

    def rot90(self,img, label):
        img = tf.image.rot90(img)

        label = tf.image.rot90(label)

        return img, label

    def crop_image(self,img, label, max_ratios=1, min_ratios=0.8, img_size=512):
        ratios = tf.random_uniform([1], maxval=max_ratios, minval=min_ratios, dtype=tf.float32)
        size = tf.cast(tf.floor(img_size * ratios), tf.int32)[0]
        offset = tf.random_uniform([2], 0, img_size + 1 - size, dtype=tf.int32)

        img = tf.image.crop_to_bounding_box(img, offset[0], offset[1], size, size)

        label = tf.image.crop_to_bounding_box(label, offset[0], offset[1], size, size)


        img = tf.image.resize_images(img, [img_size, img_size])
        label = tf.image.resize_images(label, [img_size, img_size],tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return img, label

    def original_image(self,img, label):

        return img,label

    def distorted_colors(self,image, color_ordering):
        """
        根据color_ordering的不同，对图像的亮度、对比度、颜色和饱和度进行变化
        :param image: 输入的图片
        :param color_ordering: 
        :return: 
        """
        print(color_ordering)
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
        return tf.clip_by_value(image, 0.0, 1.0)

    def data_argument(self,img, label):
        dst_image = img
        dst_label = label

        sel = tf.random_uniform([1], maxval=5, dtype=tf.int32)
        for i in range(5):
            dst_image = tf.cond(tf.equal(i, sel)[0], lambda: self.distorted_colors(dst_image, i), lambda: dst_image)


        for _ in range(3):
            index = tf.random_uniform([1], 0, 2, dtype=tf.int32)
            dst_image, dst_label = tf.cond(tf.equal(index, 0)[0],
                                                        lambda: self.rot90(dst_image, dst_label),
                                                        lambda: self.original_image(dst_image, dst_label))

        index = tf.random_uniform([1], 0, 2, dtype=tf.int32)
        dst_image, dst_label = tf.cond(tf.equal(index, 0)[0],
                                                    lambda: self.crop_image(dst_image, dst_label),
                                                    lambda: self.original_image(dst_image, dst_label))

        index = tf.random_uniform([1], 0, 2, dtype=tf.int32)
        dst_image, dst_label = tf.cond(tf.equal(index, 0)[0],
                                                    lambda: self.flip_left_right(dst_image, dst_label),
                                                    lambda: self.original_image(dst_image, dst_label))

        index = tf.random_uniform([1], 0, 2, dtype=tf.int32)
        dst_image, dst_label = tf.cond(tf.equal(index, 0)[0],
                                                    lambda: self.flip_up_dowm(dst_image, dst_label),
                                                    lambda: self.original_image(dst_image, dst_label))

        tf.summary.image('img', tf.expand_dims(img, 0))
        tf.summary.image('argu_img', tf.expand_dims(dst_image, 0))
        tf.summary.image('argu_label', tf.expand_dims(dst_label, 0))

        return dst_image, dst_label

    def split_labels(self,label):
        """
		将标签图(背景为0，云为128，雪为255)进行编码，得到三个标签图：云和背景；雪和背景；云雪和背景
		背景为0；云为1；雪为(1)2
		:param label: 
		:return: 
		"""
        label = label[..., 0]
        label = tf.cast(label * 255, tf.uint8)
        label = tf.cast(label, tf.int64)
        label=tf.expand_dims(label,axis=-1)
        cloud_mask = tf.logical_and(label > 100, label > 200)
        cloud_label = tf.where(cloud_mask, tf.ones_like(label), tf.zeros_like(label))
        return cloud_label

    def prepare_labeledData(self,data_path, data_format='*_[0-9].png', label_suffix='.png'):
        '''
		获得带标签的图像的路径及标签的路径
		要求图像与标签具有相同的文件名，存放在不同的文件夹中
		:param data_path: 图像文件的保存文件夹,nask_label(label)、reflectance_label和alpha_label在一个文件夹中
		:param data_format:图像的格式，支持通配符
		:return: 返回图像路径和标签图像路径列表
		'''
        data_list = []
        label_list = []

        data_files = glob.glob(os.path.join(data_path, data_format))
        for data_file in data_files:
            file_name = path_utils.get_filename(data_file, is_suffix=False)

            label_file = os.path.join(data_path, '%s%s' % (file_name, label_suffix))
            label_file=label_file.replace('sample','label')
            if not os.path.exists(label_file):
                print('%s label file does not exist' % file_name)
                continue


            data_list.append(data_file)
            label_list.append(label_file)

        return data_list, label_list

    def prepare_unlabeledData(self,data_path, data_format='*.png'):
        '''
		获得测试图像的路径
		:param data_path: 
		:param data_format: 
		:return: 
		'''
        data_list = []
        data_files = glob.glob(os.path.join(data_path, data_format))
        for data_file in data_files:
            data_list.append(data_file)
        return data_list

