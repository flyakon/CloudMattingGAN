'''
Cloud Mating GAN
Description: Cloud Matting Nets
Author: Wenyuan Li
Date: Feb., 2019
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from nets.cloud_gan import CloudGAN
import os
from utils import utils, path_utils
import initializer as init
from utils.dataSet_utils import MattingPrepare
import numpy  as np
from scipy import misc
from nets.vgg import vgg_16,vgg_arg_scope
from nets.resnet_v2 import resnet_v2_50,resnet_arg_scope
import time

class CloudMattingNet(object):

    def __init__(self, params:init.TrainingParamInitialization):

        self.img_size = params.img_size
        self.img_channel = params.img_channel
        self.learning_rate = params.learning_rate
        self.num_classes = params.num_classes
        self.alpha_channel = params.alpha_channel
        self.reflectance_channel = params.reflectance_channel
        self.model_path = params.model_path
        self.iter_step = params.iter_step
        self.batch_size = params.batch_size
        self.logdir = params.logdir
        self.net_name = params.net_name

        self.testDataPath = params.testDataPath
        self.result_path = params.result_path

        self.interface = self.get_interface(self.net_name)

        self.data_prepare = MattingPrepare()

    def get_interface(self,name):
        net_map = {'mattingnet': self.interface_cloudMattingNet,
                   'unet':self.interface_unet,
                   'vgg16': self.interface_vgg16,
                   'resnet50': self.interface_resnet50}

        return net_map[name]

    def fcn_arg_scope(self, weight_decay=0.0005, is_training=True, data_format='NHWC',
                      normalizer_fn =slim.batch_norm):
        """Defines the fcn arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.batch_norm],is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.fully_connected,slim.conv2d_transpose],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                normalizer_fn =normalizer_fn,
                                biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d,slim.conv2d_transpose],
                                    padding='SAME',
                                    data_format=data_format) as sc:

                    return sc

    def unet_args_cope(self, weight_decay=0.0005, data_format='NHWC'):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_regularizer=slim.l2_regularizer(weight_decay),
                            padding='VALID'):
            with slim.arg_scope([slim.max_pool2d, slim.conv2d_transpose], padding='SAME',
                                data_format=data_format) as sc:
                return sc

    def crop_and_concat(self, x1, x2):
        with tf.name_scope("crop_and_concat"):
            x1_shape = tf.shape(x1)
            x2_shape = tf.shape(x2)
            # offsets for the top left corner of the crop
            offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
            size = [-1, x2_shape[1], x2_shape[2], -1]
            x1_crop = tf.slice(x1, offsets, size)
            nets = tf.concat([x1_crop, x2], 3)
            shape2 = x2.get_shape()
            shape1 = x1.get_shape()
            nets.set_shape([self.batch_size, shape2[1], shape2[2], shape2[3] + shape1[3]])
            return nets

    def interface_cloudMattingNet(self, inputs, reuse=None, is_training=True):
        endpoints = {}
        with slim.arg_scope(self.fcn_arg_scope(is_training=is_training)):
            with tf.variable_scope('cloud_net', 'cloud_net', [inputs], reuse=reuse):
                with tf.variable_scope('feature_exatraction'):
                    nets = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    endpoints['net1'] = nets
                    nets = slim.conv2d(nets, 64, [3, 3], stride=2, scope='pool1')

                    nets = slim.repeat(nets, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    endpoints['net2'] = nets
                    nets = slim.conv2d(nets, 128, [3, 3], stride=2, scope='pool2')

                    nets = slim.repeat(nets, 2, slim.conv2d, 128, [3, 3], scope='conv3')
                    endpoints['net3'] = nets
                    nets = slim.conv2d(nets, 128, [3, 3], stride=2, scope='pool3')

                    nets = slim.repeat(nets, 2, slim.conv2d, 256, [3, 3], scope='conv4')
                    endpoints['net4'] = nets
                    nets = slim.conv2d(nets, 256, [3, 3], stride=2, scope='pool4')

                    nets = slim.repeat(nets, 2, slim.conv2d, 512, [3, 3], scope='conv5')
                    endpoints['net5'] = nets
                    nets = slim.conv2d(nets, 512, [3, 3], stride=2, scope='pool5')

                    nets = slim.repeat(nets, 2, slim.conv2d, 512, [3, 3], scope='conv6')
                    endpoints['net6'] = nets
                    nets = slim.conv2d(nets, 512, [3, 3], stride=2, scope='pool6')
                    nets = slim.conv2d(nets, 512, [3, 3], scope='conv7')
                    endpoints['net7'] = nets

                with tf.variable_scope('alpha_prediction'):
                    # alpha prediction
                    nets = endpoints['net7']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2, scope='conv_trans1') + endpoints['net6']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2, scope='conv_trans2') + endpoints['net5']
                    nets = slim.conv2d_transpose(nets, 256, [3, 3], stride=2, scope='conv_trans3') + endpoints['net4']
                    nets = slim.conv2d_transpose(nets, 128, [3, 3], stride=2, scope='conv_trans4') + endpoints['net3']
                    nets = slim.conv2d_transpose(nets, 128, [3, 3], stride=2, scope='conv_trans5') + endpoints['net2']
                    nets = slim.conv2d_transpose(nets, 64, [3, 3], stride=2, scope='conv_trans6') + endpoints['net1']
                    alpha_logits = slim.conv2d(nets, self.alpha_channel, [3, 3], scope='pred', activation_fn=None)

                with tf.variable_scope('reflectance_prediction'):
                    # reflectance prediction
                    nets = endpoints['net7']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2, scope='conv_trans1') + endpoints['net6']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2, scope='conv_trans2') + endpoints['net5']
                    nets = slim.conv2d_transpose(nets, 256, [3, 3], stride=2, scope='conv_trans3') + endpoints['net4']
                    nets = slim.conv2d_transpose(nets, 128, [3, 3], stride=2, scope='conv_trans4') + endpoints['net3']
                    nets = slim.conv2d_transpose(nets, 128, [3, 3], stride=2, scope='conv_trans5') + endpoints['net2']
                    nets = slim.conv2d_transpose(nets, 64, [3, 3], stride=2, scope='conv_trans6') + endpoints['net1']
                    reflectance_logits = slim.conv2d(nets, self.reflectance_channel, [3, 3], scope='pred',
                                                     activation_fn=None)
        return alpha_logits, reflectance_logits

    def interface_vgg16(self, inputs, reuse=None, is_training=True):
        endpoints = {}
        with slim.arg_scope(vgg_arg_scope()):
            _, vgg_end_points = vgg_16(inputs, is_training=is_training, reuse=reuse, spatial_squeeze=False,
                                       num_classes=None)

        endpoints['net1'] = vgg_end_points['vgg_16/conv1/conv1_2']
        endpoints['net2'] = vgg_end_points['vgg_16/conv2/conv2_2']
        endpoints['net3'] = vgg_end_points['vgg_16/conv3/conv3_3']
        endpoints['net4'] = vgg_end_points['vgg_16/conv4/conv4_3']
        endpoints['net5'] = vgg_end_points['vgg_16/conv5/conv5_3']

        with slim.arg_scope(self.fcn_arg_scope(is_training=is_training)):
            with tf.variable_scope('cloud_net', 'cloud_net', [inputs], reuse=reuse):
                with tf.variable_scope('feature_exatraction'):
                    nets = vgg_end_points['vgg_16/conv5/conv5_3']
                    nets = slim.conv2d(nets, 512, [3, 3], stride=2, scope='pool5')

                    nets = slim.repeat(nets, 2, slim.conv2d, 512, [3, 3], scope='conv6')
                    endpoints['net6'] = nets
                    nets = slim.conv2d(nets, 512, [3, 3], stride=2, scope='pool6')
                    nets = slim.conv2d(nets, 512, [3, 3], scope='conv7')
                    endpoints['net7'] = nets

                with tf.variable_scope('alpha_prediction'):
                    # alpha prediction
                    nets = endpoints['net7']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2, scope='conv_trans1') + endpoints['net6']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2, scope='conv_trans2') + endpoints['net5']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2, scope='conv_trans3') + endpoints['net4']
                    nets = slim.conv2d_transpose(nets, 256, [3, 3], stride=2, scope='conv_trans4') + endpoints['net3']
                    nets = slim.conv2d_transpose(nets, 128, [3, 3], stride=2, scope='conv_trans5') + endpoints['net2']
                    nets = slim.conv2d_transpose(nets, 64, [3, 3], stride=2, scope='conv_trans6') + endpoints['net1']
                    alpha_logits = slim.conv2d(nets, self.alpha_channel, [3, 3], scope='pred', activation_fn=None)

                with tf.variable_scope('reflectance_prediction'):
                    # reflectance prediction
                    nets = endpoints['net7']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2, scope='conv_trans1') + endpoints['net6']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2, scope='conv_trans2') + endpoints['net5']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2, scope='conv_trans3') + endpoints['net4']
                    nets = slim.conv2d_transpose(nets, 256, [3, 3], stride=2, scope='conv_trans4') + endpoints['net3']
                    nets = slim.conv2d_transpose(nets, 128, [3, 3], stride=2, scope='conv_trans5') + endpoints['net2']
                    nets = slim.conv2d_transpose(nets, 64, [3, 3], stride=2, scope='conv_trans6') + endpoints['net1']
                    reflectance_logits = slim.conv2d(nets, self.reflectance_channel, [3, 3], scope='pred',
                                                     activation_fn=None)
        return alpha_logits, reflectance_logits

    def interface_resnet50(self, inputs, reuse=None, is_training=False):

        endpoints = {}
        with slim.arg_scope(resnet_arg_scope(use_batch_norm=True)):
            _, resnet_endpoints = resnet_v2_50(inputs, reuse=reuse, is_training=is_training,)

        endpoints['net1'] = resnet_endpoints['resnet_v2_50/block1/unit_2/bottleneck_v2']  # 128*128 256
        endpoints['net2'] = resnet_endpoints['resnet_v2_50/block2/unit_3/bottleneck_v2']  # 64*64 512
        endpoints['net3'] = resnet_endpoints['resnet_v2_50/block3/unit_5/bottleneck_v2']  # 32*32 1024
        endpoints['net4'] = resnet_endpoints['resnet_v2_50/block4/unit_3/bottleneck_v2']  # 16*16 2048

        with slim.arg_scope(self.fcn_arg_scope(is_training=is_training,normalizer_fn=None)):
            with tf.variable_scope('cloud_net', 'cloud_net', [inputs], reuse=reuse):

                with tf.variable_scope('alpha_prediction'):
                    # alpha prediction
                    nets = resnet_endpoints['resnet_v2_50/block4']  # 64*64*2048
                    nets = slim.conv2d_transpose(nets, 512, kernel_size=[3, 3], stride=2) + resnet_endpoints[
                        'resnet_v2_50/block2/unit_2/bottleneck_v2']

                    nets = slim.conv2d_transpose(nets, 256, kernel_size=[3, 3], stride=2) + resnet_endpoints[
                        'resnet_v2_50/block1/unit_2/bottleneck_v2']

                    nets = slim.conv2d_transpose(nets, 64, kernel_size=[3, 3], stride=2) + resnet_endpoints[
                        'resnet_v2_50/conv1']

                    alpha_logits = slim.conv2d(nets, self.alpha_channel, [3, 3], scope='pred', activation_fn=None)

                with tf.variable_scope('reflectance_prediction'):
                    # reflectance prediction
                    nets = resnet_endpoints['resnet_v2_50/block4']  # 64*64*2048
                    nets = slim.conv2d_transpose(nets, 512, kernel_size=[3, 3], stride=2) + resnet_endpoints[
                        'resnet_v2_50/block2/unit_2/bottleneck_v2']

                    nets = slim.conv2d_transpose(nets, 256, kernel_size=[3, 3], stride=2) + resnet_endpoints[
                        'resnet_v2_50/block1/unit_2/bottleneck_v2']

                    nets = slim.conv2d_transpose(nets, 64, kernel_size=[3, 3], stride=2) + resnet_endpoints[
                        'resnet_v2_50/conv1']

                    reflectance_logits = slim.conv2d(nets, self.reflectance_channel, [3, 3], scope='pred',
                                                     activation_fn=None)
        return alpha_logits, reflectance_logits

    def interface_unet(self, inputs, reuse=None, is_training=True):
        endpoints = {}
        with slim.arg_scope(self.fcn_arg_scope(is_training=is_training)):
            with tf.variable_scope('cloud_net', 'cloud_net', [inputs], reuse=reuse):
                with tf.variable_scope('feature_exatraction'):
                    nets = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3])  # 508*508*64
                    endpoints['net1'] = nets
                    nets = slim.max_pool2d(nets, [2, 2])  # 254*254*64

                    nets = slim.repeat(nets, 2, slim.conv2d, 128, [3, 3])  # 250*250*128
                    endpoints['net2'] = nets
                    nets = slim.max_pool2d(nets, [2, 2])  # 125*125*128

                    nets = slim.repeat(nets, 2, slim.conv2d, 256, [3, 3])  # 121*121*256
                    endpoints['net3'] = nets
                    nets = slim.max_pool2d(nets, [2, 2])  # 61*61*256

                    nets = slim.repeat(nets, 2, slim.conv2d, 512, [3, 3])  # 57*57*512
                    endpoints['net4'] = nets
                    nets = slim.max_pool2d(nets, [2, 2])  # 29*29*512

                    nets = slim.repeat(nets, 2, slim.conv2d, 1024, [3, 3])  # 25*25*1024
                    endpoints['net5'] = nets

                with tf.variable_scope('alpha_prediction'):
                    nets=endpoints['net5']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2)  # 50*50*512
                    nets = self.crop_and_concat(endpoints['net4'], nets)
                    nets = slim.repeat(nets, 2, slim.conv2d, 512, [3, 3])  # 46*46*512

                    nets = slim.conv2d_transpose(nets, 256, [3, 3], stride=2)  # 92*92*256
                    nets = self.crop_and_concat(endpoints['net3'], nets)
                    nets = slim.repeat(nets, 2, slim.conv2d, 256, [3, 3])  # 88*88*256

                    nets = slim.conv2d_transpose(nets, 128, [3, 3], stride=2)  # 176*176*128
                    nets = self.crop_and_concat(endpoints['net2'], nets)
                    nets = slim.repeat(nets, 2, slim.conv2d, 128, [3, 3])  # 172*172*128

                    nets = slim.conv2d_transpose(nets, 64, [3, 3], stride=2)  # 344*344*64
                    nets = self.crop_and_concat(endpoints['net1'], nets)
                    nets = slim.repeat(nets, 2, slim.conv2d, 64, [3, 3])  # 340*340*64

                    logits = slim.conv2d(nets, self.alpha_channel, [3, 3], padding='SAME',activation_fn=None)
                    alpha_logits = tf.image.resize_images(logits, [self.img_size, self.img_size])

                with tf.variable_scope('reflectance_prediction'):
                    nets = endpoints['net5']
                    nets = slim.conv2d_transpose(nets, 512, [3, 3], stride=2)  # 50*50*512
                    nets = self.crop_and_concat(endpoints['net4'], nets)
                    nets = slim.repeat(nets, 2, slim.conv2d, 512, [3, 3])  # 46*46*512

                    nets = slim.conv2d_transpose(nets, 256, [3, 3], stride=2)  # 92*92*256
                    nets = self.crop_and_concat(endpoints['net3'], nets)
                    nets = slim.repeat(nets, 2, slim.conv2d, 256, [3, 3])  # 88*88*256

                    nets = slim.conv2d_transpose(nets, 128, [3, 3], stride=2)  # 176*176*128
                    nets = self.crop_and_concat(endpoints['net2'], nets)
                    nets = slim.repeat(nets, 2, slim.conv2d, 128, [3, 3])  # 172*172*128

                    nets = slim.conv2d_transpose(nets, 64, [3, 3], stride=2)  # 344*344*64
                    nets = self.crop_and_concat(endpoints['net1'], nets)
                    nets = slim.repeat(nets, 2, slim.conv2d, 64, [3, 3])  # 340*340*64

                    logits = slim.conv2d(nets, self.reflectance_channel, [3, 3], padding='SAME',activation_fn=None)
                    reflectance_logits = tf.image.resize_images(logits, [self.img_size, self.img_size])

            return alpha_logits,reflectance_logits

    def calc_losses(self, alpha_logits, reflectance_loggits, alpha, reflectance):

        alpha_loss = tf.abs(alpha_logits-alpha)
        alpha_loss = tf.reduce_mean(alpha_loss)
        # alpha_loss = tf.reduce_mean(alpha_loss)
        tf.summary.scalar('alpha_loss', alpha_loss)
        reflectance_loss = tf.abs(reflectance_loggits - reflectance)
        reflectance_loss = tf.reduce_mean(reflectance_loss)
        # reflectance_loss = tf.reduce_mean(reflectance_loss)
        tf.summary.scalar('reflectance_loss', reflectance_loss)
        total_loss = alpha_loss+reflectance_loss
        tf.summary.scalar('total_loss', total_loss)

        return total_loss

    def train_op(self, large_image, alpha, reflectance):

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            global_step = tf.train.get_or_create_global_step()

            learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate, global_step=global_step,
                                                       decay_rate=0.96, decay_steps=10000)
            alpha_logits, reflectance_logits = self.interface(large_image)
            loss = self.calc_losses(alpha_logits, reflectance_logits, alpha, reflectance)

            optimizer = tf.train.AdamOptimizer(learning_rate)\
                .minimize(loss,var_list=slim.get_trainable_variables('cloud_net'),global_step=global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                loss = control_flow_ops.with_dependencies([updates], loss)
            tf.summary.scalar('learning_rate', learning_rate)

        return optimizer, loss, global_step


    def test_op(self, large_image, reuse=None):

        alpha_logits, reflectance_logits = self.interface(large_image, reuse=reuse, is_training=False)

        mask = alpha_logits < 0
        alpha_logits = tf.where(mask, tf.zeros_like(alpha_logits), alpha_logits)
        mask = alpha_logits > 1
        alpha_logits = tf.where(mask, tf.ones_like(alpha_logits), alpha_logits)

        mask = reflectance_logits > 1
        reflectance_logits = tf.where(mask, tf.ones_like(reflectance_logits), reflectance_logits)
        mask = reflectance_logits < 0
        reflectance_logits = tf.where(mask, tf.zeros_like(reflectance_logits), reflectance_logits)
        return [large_image[0], tf.cast(alpha_logits[0]*255, tf.uint8),
                tf.cast(reflectance_logits[0] * 255, tf.uint8)]

    def validate_op(self, large_image, alpha, reflectance, reuse=None):

        alpha_logits, reflectance_logits = self.interface(large_image, reuse=reuse, is_training=False)
        loss = self.calc_losses(alpha_logits, reflectance_logits, alpha, reflectance)

        mask = alpha_logits < 0
        alpha_logits = tf.where(mask, tf.zeros_like(alpha_logits), alpha_logits)
        mask = alpha_logits > 1
        alpha_logits = tf.where(mask, tf.ones_like(alpha_logits), alpha_logits)

        mask = reflectance_logits > 1
        reflectance_logits = tf.where(mask, tf.ones_like(reflectance_logits), reflectance_logits)
        mask = reflectance_logits < 0
        reflectance_logits = tf.where(mask, tf.zeros_like(reflectance_logits), reflectance_logits)

        return [large_image[0], tf.cast(alpha_logits[0]*255, tf.uint8),
               tf.cast(reflectance_logits[0] * 255, tf.uint8), loss]


    def prepare_test_data(self):

        testImage_list = self.data_prepare.prepare_unlabeledData(data_path=self.testDataPath, data_format='*.jpg')

        testImage_producer = tf.train.slice_input_producer([testImage_list], shuffle=False,
                                                           capacity=self.batch_size * 5, num_epochs=1)
        image_file = tf.read_file(testImage_producer[0])

        testImage = tf.image.decode_png(image_file, channels=self.img_channel)
        testImage = tf.cast(testImage, tf.float32) / 255.
        testImage = tf.image.resize_images(testImage, [self.img_size, self.img_size])
        testImage.set_shape([self.img_size, self.img_size, self.img_channel])

        self.test_image = tf.expand_dims(testImage, axis=0)
        self.test_name = testImage_producer[0]

    def load_checkpoints(self, sess):

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if 'checkpoint' in os.listdir(self.model_path):
            print('restore from last model....')
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('restore ' + ckpt.model_checkpoint_path)
            else:
                print('load model error')
        else:
            self.gan.load_historical_model(sess)
            if 'vgg' in self.net_name:
                model_file = os.path.join('model', 'vgg_16.ckpt')
                if not os.path.exists(model_file):
                    print('no vgg16 model found, start training from scratch!')
                else:
                    print('restore partial model vgg16')
                    variables_to_restore = utils.get_variables_to_restore(['vgg_16'], ['Adam', 'Adam_1'])
                    saver = tf.train.Saver(variables_to_restore)
                    saver.restore(sess, model_file)
            elif 'resnet' in self.net_name:
                model_file = os.path.join('model', 'resnet_v2_50.ckpt')
                if not os.path.exists(model_file):
                    print('no resnet50 model found, start training from scratch!')
                else:
                    print('restore partial model resnet_v2_50')
                    variables_to_restore = utils.get_variables_to_restore(['resnet_v2_50'], ['Adam', 'Adam_1'])
                    saver = tf.train.Saver(variables_to_restore)
                    saver.restore(sess, model_file)

    def run_train_loop(self):

        params = init.TrainingParamInitialization()
        self.gan = CloudGAN(params)
        self.train_img, self.train_alpha, self.train_reflectance =\
            self.data_prepare.preprocess(self.gan.G_sample, self.gan.G_alpha, self.gan.G_relectance)
        self.validate_img, self.validate_alpha, self.validate_reflectance = \
            self.data_prepare.preprocess(self.gan.G_sample, self.gan.G_alpha,self.gan.G_relectance)

        print('build matting net')
        with tf.name_scope('train'):
            train_op = self.train_op(self.train_img, self.train_alpha, self.train_reflectance)
        with tf.name_scope('validate'):
            validate_op = self.validate_op(self.validate_img, self.validate_alpha, self.validate_reflectance, reuse=True)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.load_checkpoints(sess)
        step = 0
        log_step = 50
        saver = tf.train.Saver()
        merge_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        if not os.path.exists('rst'):
            os.mkdir('rst')
        while step < self.iter_step:

                feed_dict = None

                self.gan.run_train(sess)
                X_mb = utils.get_batch(self.gan.data, self.batch_size, 'cloudimage', self.img_size)
                Z_mb = utils.get_batch(self.gan.data, self.batch_size, 'cloudimage', self.img_size)
                BG_mb = utils.get_batch(self.gan.data, self.batch_size, 'background', self.img_size)
                feed_dict = {self.gan.X: X_mb, self.gan.Z: Z_mb, self.gan.BG: BG_mb}

                [_, train_loss, step] = sess.run(train_op, feed_dict=feed_dict)

                if np.mod(step, log_step) == 1:

                    X_mb = utils.get_batch(self.gan.data, self.batch_size, 'cloudimage', self.img_size)
                    Z_mb = utils.get_batch(self.gan.data, self.batch_size, 'cloudimage', self.img_size)
                    BG_mb = utils.get_batch(self.gan.data, self.batch_size, 'background', self.img_size)
                    feed_dict = {self.gan.X: X_mb, self.gan.Z: Z_mb, self.gan.BG: BG_mb}

                    merges = [merge_op] + validate_op

                    summary, image, alpha_image, reflectance_image, validate_loss = sess.run(merges,
                                                                                feed_dict=feed_dict)

                    summary_writer.add_summary(summary, step)
                    print("step:%d,loss:%f,validate_loss:%f" % (step, train_loss, validate_loss))
                    misc.imsave('rst/' + str(step) + '_image.png', image)
                    misc.imsave('rst/' + str(step) + '_alpha.png', alpha_image)
                    misc.imsave('rst/' + str(step) + '_foreground.png', reflectance_image)

                if np.mod(step, 500) == 1:
                    saver.save(sess, os.path.join(self.model_path, 'model'), step)
                    
    def run_test_loop(self):

        print('prepare test data')
        self.prepare_test_data()

        print('building graphics')
        test_op = self.test_op(self.test_image)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            self.load_checkpoints(sess)

            testResult_Folder = self.result_path
            if not os.path.exists(testResult_Folder):
                os.mkdir(testResult_Folder)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                i = 0
                start = time.time()
                while not coord.should_stop():
                    image, alpha, matting_image, filename = sess.run(test_op + [self.test_name])
                    filename = filename.decode('utf-8')
                    filename = path_utils.get_filename(filename, is_suffix=False)
                    filename = filename.replace('_img', '')

                    misc.imsave(os.path.join(testResult_Folder, filename + '_image.png'), image)
                    misc.imsave(os.path.join(testResult_Folder, filename + '_alpha.png'), alpha)
                    misc.imsave(os.path.join(testResult_Folder, filename + '_reflectance.png'), matting_image)

                    print('processing %dth image:%s' % (i, filename))
                    i += 1
            except tf.errors.OutOfRangeError:
                print('training completed')
            finally:
                coord.request_stop()
                end = time.time()
                print('test completely!')
                print('time:%f' % (end - start))
            coord.join(threads=threads)

