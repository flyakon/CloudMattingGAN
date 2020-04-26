'''
Cloud Mating GAN
Description: Parameters Configuration
Author: Wenyuan Li
Date: Feb., 2019
'''
import tensorflow as tf

#  Configuration parameters for Matting or Segment


tf.app.flags.DEFINE_string('testDataPath',r'dataset\test\thin_slice','test data path for matting')
tf.app.flags.DEFINE_string('model_path',r'model\cloud_mating_model','')
tf.app.flags.DEFINE_string('logdir',r'log\cloud_mating_log','')
tf.app.flags.DEFINE_string('net_name','unet','mattingnet,unet,vgg16 or reset50')
tf.app.flags.DEFINE_string('result_path',r'test_result','')

tf.app.flags.DEFINE_integer('batch_size',2,'')
tf.app.flags.DEFINE_integer('img_size',512,'')
tf.app.flags.DEFINE_integer('num_classes',2,'')
tf.app.flags.DEFINE_float('learning_rate',1e-4,'training learning rate')
tf.app.flags.DEFINE_integer('img_channel',3,'')
tf.app.flags.DEFINE_integer('alpha_channel',3,'')
tf.app.flags.DEFINE_integer('reflectance_channel',3,'')
tf.app.flags.DEFINE_integer('iter_step',30002,'')


#  Configuration parameters for GAN
tf.app.flags.DEFINE_integer('D_input_size',128,'')
tf.app.flags.DEFINE_integer('G_input_size',128,'')
tf.app.flags.DEFINE_float('g_learning_rate',1e-6,'training learning rate')
tf.app.flags.DEFINE_float('d_learning_rate',1e-7,'training learning rate')
tf.app.flags.DEFINE_string('gan_model','LSGAN','W_GAN Vanilla_GAN和LSGAN')

tf.app.flags.DEFINE_string('image_dir',r'dataset\train',
                           'images to train GAN or CloudMatting')
tf.app.flags.DEFINE_string('checkpoint_gan',r'model\LSGAN',
                           'checkpoint path for GAN')
tf.app.flags.DEFINE_string('sample_dir',r'samples',
                           'training result path for GAN or CloudMatting')
tf.app.flags.DEFINE_string('result_dir',r'result',
                           'Directory name to save the generated images,only for cloud_gereration.py')

FLAGS=tf.app.flags.FLAGS

