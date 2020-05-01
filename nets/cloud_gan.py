'''
Cloud Mating GAN
Description: Cloud Matting Nets
Author: Zhengxia Zou and Wenyuan Li
Date: Feb., 2019
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim

import os
import matplotlib.pylab as plt

import initializer as init
import numpy as np
from utils import utils


class CloudGAN(object):


    def __init__(self,params:init.TrainingParamInitialization):

        self.img_size = params.img_size
        self.img_channel = params.img_channel
        self.alpha_channel = params.alpha_channel
        self.reflectance_channel = params.reflectance_channel

        self.batch_size = params.batch_size
        self.D_input_size = params.D_input_size
        self.G_input_size = params.G_input_size

        self.image_dir = params.image_dir
        self.checkpoint_gan = params.checkpoint_gan
        self.sample_dir = params.sample_dir

        self.result_dir=params.result_dir


        self.g_learning_rate = params.g_learning_rate
        self.d_learning_rate = params.d_learning_rate
        self.d_clip = params.d_clip         #生成器梯度限幅
        self.gan_model = params.gan_model   #gan的类型valinila、WGAN和LSGAN
        self.optimizer=params.optimizer



        self.X = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.img_channel])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.img_channel])
        self.BG = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.img_channel])

        self.mm=1

        self.iter_step=params.iter_step
        self.data = utils.load_images(self.image_dir, self.img_size)

        print('start building GAN graphics...')
        self.build_graphics()



    def generator(self,Z, BG,reuse=None):
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('Generator_scope'):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                weights_initializer = tf.contrib.layers.xavier_initializer(),
                                normalizer_fn=slim.batch_norm
                                ):

                Z_ = tf.image.resize_images(images=Z, size=[self.G_input_size, self.G_input_size])
                BG_ = tf.image.resize_images(images=BG, size=[self.G_input_size, self.G_input_size])
                Z_ = tf.concat([Z_, BG_], axis=-1)

                # encoder
                net_a_1 = slim.repeat(
                    Z_, 2, slim.conv2d, 64, [3, 3], scope='conv1_ea')
                # net1 = slim.dropout(net1)
                net_a_1 = slim.max_pool2d(net_a_1, [2, 2], scope='pool1_ea')

                net_a_2 = slim.repeat(
                    net_a_1, 2, slim.conv2d, 64, [3, 3], scope='conv2_ea')
                # net2 = slim.dropout(net2)
                net_a_2 = slim.max_pool2d(net_a_2, [2, 2], scope='pool2_ea')

                net_a_3 = slim.repeat(
                    net_a_2, 2, slim.conv2d, 64, [3, 3], scope='conv3_ea')
                net_a_3 = slim.max_pool2d(net_a_3, [2, 2], scope='pool3_ea')

                net_a_4 = slim.repeat(
                    net_a_3, 2, slim.conv2d, 64, [3, 3], scope='conv4_ea')
                net4 = slim.max_pool2d(net_a_4, [2, 2], scope='pool4_ea')

                # decoder
                net_1 = slim.repeat(
                    net4, 2, slim.conv2d, 64, [5, 5], scope='conv1_d1')
                # net_1 = slim.conv2d_transpose(net_1, 256, [5, 5], stride=2)
                net_1 = tf.image.resize_images(images=net_1, size=[int(self.G_input_size/8), int(self.G_input_size/8)])
                net_1 = net_1 + net_a_3

                net_2 = slim.repeat(
                    net_1, 2, slim.conv2d, 64, [5, 5], scope='conv2_d1')
                # net_2 = slim.conv2d_transpose(net_2, 128, [5, 5], stride=2)
                net_2 = tf.image.resize_images(images=net_2, size=[int(self.G_input_size/4), int(self.G_input_size/4)])
                net_2 = net_2 + net_a_2

                net_3 = slim.repeat(
                    net_2, 2, slim.conv2d, 64, [5, 5], scope='conv3_d1')
                # net_3 = slim.conv2d_transpose(net_3, 64, [5, 5], stride=2)
                net_3 = tf.image.resize_images(images=net_3, size=[int(self.G_input_size/2), int(self.G_input_size/2)])
                net_3 = net_3 + net_a_1

                net_4 = slim.repeat(
                    net_3, 2, slim.conv2d, 64, [5, 5], scope='conv4_d1')
                # net_4 = slim.conv2d_transpose(net_4, 64, [5, 5], stride=2)
                net_4 = tf.image.resize_images(images=net_4, size=[int(self.G_input_size), int(self.G_input_size)])

                reflectance_ = slim.conv2d(net_4, 3, [3, 3], activation_fn=None)
                reflectance_ = tf.nn.sigmoid(reflectance_)

                # build high reso maps
                reflectance = tf.image.resize_images(images=reflectance_, size=[self.img_size, self.img_size])
                reflectance = reflectance * Z * 2
                reflectance = tf.clip_by_value(reflectance, clip_value_min=0, clip_value_max=1.)

                alpha = reflectance * (0.2*tf.random_uniform([self.batch_size, 1, 1, 1]) + 1.)
                alpha = tf.clip_by_value(alpha, clip_value_min=0, clip_value_max=1.)

                sample = reflectance + (1 - alpha) * BG
                # sample = tf.clip_by_value(sample, clip_value_min=0, clip_value_max=1.)

        return sample, reflectance, alpha




    def discriminator(self,x, reuse=False):
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('Discriminator_scope'):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                normalizer_fn=slim.batch_norm,
                                # weights_regularizer=slim.l2_regularizer(0.01)
                                ):

                x = tf.image.resize_images(images=x, size=[self.D_input_size, self.D_input_size])

                net = slim.repeat(
                    x, 2, slim.conv2d, 128, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')

                net = slim.repeat(
                    net, 2, slim.conv2d, 256, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')

                net = slim.repeat(
                    net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')

                net = slim.repeat(
                    net, 2, slim.conv2d, 256, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')

                net = slim.repeat(
                    net, 2, slim.conv2d, 256, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                net = slim.flatten(net)

                net = slim.fully_connected(net, 512)
                D_logit = slim.fully_connected(net, 1, activation_fn=None)
                D_prob = tf.nn.sigmoid(D_logit)

        return D_logit, D_prob

    def calc_loss(self,D_logits_real,D_logits_fake,G_relectance):

        max_rgb_value = tf.reduce_max(G_relectance, axis=-1)
        min_rgb_value = tf.reduce_min(G_relectance, axis=-1)
        S_value = (max_rgb_value - min_rgb_value) / (max_rgb_value + 1e-3)
        penalty = 1. * tf.norm(S_value)
        #### WGAN
        if self.gan_model is 'W_GAN':

            D_loss = - (tf.reduce_mean(D_logits_real) - tf.reduce_mean(D_logits_fake))
            G_loss = - tf.reduce_mean(D_logits_fake)+penalty
            tf.summary.scalar('D_loss', D_loss)
            tf.summary.scalar('G_loss', G_loss)
        #### Vanilla_GAN
        elif self.gan_model=='Vanilla_GAN':

            D_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_logits_real)))
            D_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.zeros_like(D_logits_fake)))
            D_loss = D_loss_real + D_loss_fake
            G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                labels=tf.ones_like(D_logits_fake))) + penalty
            tf.summary.scalar('D_loss', D_loss)
            tf.summary.scalar('G_loss', G_loss)
        else:
            D_loss = 0.5 * (tf.reduce_mean((D_logits_real - 1)**2) + tf.reduce_mean(D_logits_fake**2))
            G_loss = 0.5 * tf.reduce_mean((D_logits_fake - 1)**2)+penalty
            tf.summary.scalar('D_loss', D_loss)
            tf.summary.scalar('G_loss', G_loss)
        return G_loss,D_loss

    def load_historical_model(self,sess):

        # check and create model dir
        if os.path.exists(self.checkpoint_gan) is False:
            os.makedirs(self.checkpoint_gan)

        if 'checkpoint' in os.listdir(self.checkpoint_gan):
            # training from the last checkpoint
            print('loading model from the last checkpoint ...')
            saver = tf.train.Saver(
                utils.get_variables_to_restore(['Generator_scope', 'Discriminator_scope', 'G_steps', 'D_steps'],
                                               ['Adam', 'Adam_1']))
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_gan)
            saver.restore(sess, latest_checkpoint)
            print(latest_checkpoint)
            print('loading finished!')
        else:
            print('no historical gan model found, start training from scratch!')

    def run_train_loop(self):

        G_sample, G_relectance, G_alpha = self.generator(self.Z, self.BG)
        D_logits_real, D_prob_real = self.discriminator(self.X)
        D_logits_fake, D_prob_fake = self.discriminator(G_sample, reuse=True)
        G_loss,D_loss=self.calc_loss(D_logits_real,D_logits_fake,G_relectance)

        tvars = tf.trainable_variables()
        theta_D = [var for var in tvars if 'Discriminator_scope' in var.name]
        theta_G = [var for var in tvars if 'Generator_scope' in var.name]

        # to record the iteration number
        G_steps = tf.Variable(0, trainable=False, name='G_steps')
        D_steps = tf.Variable(0, trainable=False, name='D_steps')

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            if self.optimizer is 'RMSProp':
                D_solver = (tf.train.RMSPropOptimizer(learning_rate=self.d_learning_rate).minimize(D_loss, var_list=theta_D,
                                                                                              global_step=D_steps))
                G_solver = (tf.train.RMSPropOptimizer(learning_rate=self.g_learning_rate).minimize(G_loss, var_list=theta_G,
                                                                                              global_step=G_steps))

            if self.optimizer is 'Adam':
                D_solver = (tf.train.AdamOptimizer(learning_rate=self.d_learning_rate).minimize(D_loss, var_list=theta_D,
                                                                                           global_step=D_steps))
                G_solver = (tf.train.AdamOptimizer(learning_rate=self.g_learning_rate).minimize(G_loss, var_list=theta_G,
                                                                                           global_step=G_steps))

            clip_D = [p.assign(tf.clip_by_value(p, -self.d_clip, self.d_clip)) for p in theta_D]

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # create a summary writer
        merged_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('temp', sess.graph)

        # load historical model and create a saver
        self.load_historical_model(sess)
        saver = tf.train.Saver()

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)


        g_iter = 0
        d_iter = 0
        mm = 1

        while g_iter < self.iter_step:

            X_mb = utils.get_batch(self.data, self.batch_size, 'cloudimage', self.img_size)
            Z_mb = utils.get_batch(self.data, self.batch_size, 'cloudimage', self.img_size)
            BG_mb = utils.get_batch(self.data, self.batch_size, 'background', self.img_size)

            _, D_loss_curr, _, summary = sess.run([D_solver, D_loss, clip_D, merged_op],
                                                  feed_dict={self.X: X_mb, self.Z: Z_mb, self.BG: BG_mb})
            d_iter = sess.run(D_steps)
            # write states to summary
            summary_writer.add_summary(summary, g_iter)

            if D_loss_curr < 0.1:
                mm = mm + 5
            else:
                mm = 1

            for _ in range(mm):
                X_mb = utils.get_batch(self.data, self.batch_size, 'cloudimage', self.img_size)
                Z_mb = utils.get_batch(self.data, self.batch_size, 'cloudimage', self.img_size)
                BG_mb = utils.get_batch(self.data, self.batch_size, 'background', self.img_size)
                _, G_loss_curr, D_loss_curr = sess.run([G_solver, G_loss, D_loss],
                                                       feed_dict={self.X: X_mb, self.Z: Z_mb, self.BG: BG_mb})
                g_iter = sess.run(G_steps)

                # save generated samples
                if g_iter % 5 == 0:
                    samples, reflectance, alpha = sess.run([G_sample, G_relectance, G_alpha],
                        feed_dict={self.Z: Z_mb, self.BG: BG_mb})


                    samples=np.array(samples*255,np.uint8)
                    reflectance=np.array(reflectance*255,np.uint8)
                    alpha=np.array(alpha*255,np.uint8)
                    BG=np.array(BG_mb*255,np.uint8)
                    for jj in range(self.batch_size):
                        i=g_iter
                        save_path = os.path.join(self.sample_dir,
                                                 '{}'.format(str(i).zfill(5)) + '_' + str(jj) + '.jpg')
                        plt.imsave(save_path, samples[jj, :, :, :])

                        save_path = os.path.join(self.sample_dir,
                                                 '{}'.format(str(i).zfill(5)) + '_' + str(jj) + '_reflectance.png')
                        plt.imsave(save_path, reflectance[jj, :, :, :])

                        save_path = os.path.join(self.sample_dir,'{}'.format(str(i).zfill(5)) + '_' + str(jj) + '_alpha.png')
                        plt.imsave(save_path, alpha[jj, :, :, :])
                        save_path = os.path.join(self.sample_dir,'{}'.format(str(i).zfill(5)) + '_' + str(jj) + '_BG.png')
                        plt.imsave(save_path, BG[jj, :, :, :])

                if g_iter % 50 == 0:
                    print('D_loss = %g, G_loss = %g' % (D_loss_curr, G_loss_curr))
                    print('g_iter = %d, d_iter = %d, n_g/d = %d' % (g_iter, d_iter, mm))


                # save model every 500 g_iters
                if np.mod(g_iter, 500) == 1 and g_iter > 1:
                    print('saving model to checkpoint ...')
                    saver.save(sess, os.path.join(self.checkpoint_gan, 'G_step'), global_step=G_steps)



    def build_graphics(self):

        self.G_sample, self.G_relectance, self.G_alpha = self.generator(self.Z, self.BG)
        self.D_logits_real, self.D_prob_real = self.discriminator(self.X)
        self.D_logits_fake, self.D_prob_fake = self.discriminator(self.G_sample, reuse=True)
        self.G_loss, self.D_loss = self.calc_loss(self.D_logits_real, self.D_logits_fake, self.G_relectance)

        tvars = tf.trainable_variables()
        theta_D = [var for var in tvars if 'Discriminator_scope' in var.name]
        theta_G = [var for var in tvars if 'Generator_scope' in var.name]

        # to record the iteration number
        self.G_steps = tf.Variable(0, trainable=False, name='G_steps')
        self.D_steps = tf.Variable(0, trainable=False, name='D_steps')

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            if self.optimizer is 'RMSProp':
                self.D_solver = (
                tf.train.RMSPropOptimizer(learning_rate=self.d_learning_rate).minimize(self.D_loss, var_list=theta_D,
                                                                                       global_step=self.D_steps))
                self.G_solver = (
                tf.train.RMSPropOptimizer(learning_rate=self.g_learning_rate).minimize(self.G_loss, var_list=theta_G,
                                                                                       global_step=self.G_steps))

            if self.optimizer is 'Adam':
                self.D_solver = (
                tf.train.AdamOptimizer(learning_rate=self.d_learning_rate).minimize(self.D_loss, var_list=theta_D,
                                                                                    global_step=self.D_steps))
                self.G_solver = (
                tf.train.AdamOptimizer(learning_rate=self.g_learning_rate).minimize(self.G_loss, var_list=theta_G,
                                                                                    global_step=self.G_steps))

            self.clip_D = [p.assign(tf.clip_by_value(p, -self.d_clip, self.d_clip)) for p in theta_D]


    def run_train(self,sess):

        X_mb = utils.get_batch(self.data, self.batch_size, 'cloudimage', self.img_size)
        Z_mb = utils.get_batch(self.data, self.batch_size, 'cloudimage', self.img_size)
        BG_mb = utils.get_batch(self.data, self.batch_size, 'background', self.img_size)

        _, D_loss_curr, _ = sess.run([self.D_solver, self.D_loss, self.clip_D],
                                     feed_dict={self.X: X_mb, self.Z: Z_mb, self.BG: BG_mb})
        d_iter = sess.run(self.D_steps)

        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

        if D_loss_curr < 0.1:
            self.mm = self.mm + 1
        else:
            self.mm = 1

        for _ in range(self.mm):
            X_mb = utils.get_batch(self.data, self.batch_size, 'cloudimage', self.img_size)
            Z_mb = utils.get_batch(self.data, self.batch_size, 'cloudimage', self.img_size)
            BG_mb = utils.get_batch(self.data, self.batch_size, 'background', self.img_size)

            _, G_loss_curr, D_loss_curr = sess.run([self.G_solver, self.G_loss, self.D_loss],
                                                   feed_dict={self.X: X_mb, self.Z: Z_mb, self.BG: BG_mb})
            g_iter = sess.run(self.G_steps)

            # save generated samples
            if g_iter % 50 == 0:
                samples, reflectance, alpha = sess.run([self.G_sample, self.G_relectance, self.G_alpha],
                                                       feed_dict={self.Z: Z_mb, self.BG: BG_mb})

                save_path = os.path.join(self.sample_dir, '{}_reflectance.png'.format(str(g_iter).zfill(5)))
                plt.imsave(save_path, utils.plot2x2(reflectance), vmax=1, vmin=0)

                save_path = os.path.join(self.sample_dir, '{}_image.png'.format(str(g_iter).zfill(5)))
                plt.imsave(save_path, utils.plot2x2(samples), vmax=1, vmin=0)

                save_path = os.path.join(self.sample_dir, '{}_input.png'.format(str(g_iter).zfill(5)))
                plt.imsave(save_path, utils.plot2x2(Z_mb), vmax=1, vmin=0)

                save_path = os.path.join(self.sample_dir, '{}_alpha.png'.format(str(g_iter).zfill(5)))
                plt.imsave(save_path, utils.plot2x2(alpha), vmax=1, vmin=0)

            if g_iter % 50 == 0:
                print('D_loss = %g, G_loss = %g' % (D_loss_curr, G_loss_curr))
                print('g_iter = %d, d_iter = %d, n_g/d = %d' % (g_iter, d_iter, self.mm))
                print()


    def run_test_loop(self):
        G_sample, G_relectance, G_alpha = self.generator(self.Z, self.BG)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # load the detection model
        self.load_historical_model(sess)

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        for i in range(self.iter_step):
            BG_mb = utils.get_batch(self.data, self.batch_size, mode='background',
                                    with_data_augmentation=True, img_size=self.img_size)
            Z_mb = utils.get_batch(self.data, self.batch_size, mode='cloudimage',
                                   with_data_augmentation=True, img_size=self.img_size)

            sample, reflectance, alpha = sess.run([G_sample, G_relectance, G_alpha],
                                                  feed_dict={self.Z: Z_mb, self.BG: BG_mb})

            for jj in range(self.batch_size):
                save_path = os.path.join(self.result_dir, '{}'.format(str(i).zfill(5)) + '_' + str(jj) + '_image.jpg')
                plt.imsave(save_path, sample[jj, :, :, :])

                save_path = os.path.join(self.result_dir, '{}'.format(str(i).zfill(5)) + '_' + str(jj) + '_reflectance.png')
                plt.imsave(save_path, reflectance[jj, :, :, :])

                save_path = os.path.join(self.result_dir, '{}'.format(str(i).zfill(5)) + '_' + str(jj) + '_alpha.png')
                plt.imsave(save_path, alpha[jj, :, :, :])

