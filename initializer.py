"""
Cloud Generation
Description: initializer
Author: Zhengxia Zou
Date: Feb., 2019
"""

from config import FLAGS

class TrainingParamInitialization():
    """Initialize all params for training and detection"""

    def __init__(self):

        self.img_size = FLAGS.img_size
        self.alpha_channel = FLAGS.alpha_channel
        self.reflectance_channel = FLAGS.reflectance_channel
        self.img_channel = FLAGS.img_channel

        self.D_input_size = FLAGS.D_input_size
        self.G_input_size = FLAGS.G_input_size
        self.batch_size = FLAGS.batch_size

        self.g_learning_rate = FLAGS.g_learning_rate
        self.d_learning_rate = FLAGS.d_learning_rate

        self.gan_model = FLAGS.gan_model
        if self.gan_model == 'Vanilla_GAN':
            self.d_clip=1e9
            self.optimizer = 'Adam'
        elif self.gan_model == 'W_GAN':
            self.d_clip = 0.05
            self.optimizer = 'RMSProp'
        elif self.gan_model == 'LSGAN':
            self.d_clip = 0.05
            self.optimizer = 'RMSProp'
        else:
            print('gan model does not exist!')
            exit(-1)

        self.image_dir = FLAGS.image_dir
        self.checkpoint_gan = FLAGS.checkpoint_gan # Directory name to save the checkpoints
        self.sample_dir = FLAGS.sample_dir # Directory name to save the samples on training
        self.result_dir = FLAGS.result_dir # Directory name to save the generated images

        self.iter_step = FLAGS.iter_step        # number of training iterations

        self.learning_rate = FLAGS.learning_rate
        self.num_classes = FLAGS.num_classes

        self.testDataPath = FLAGS.testDataPath
        self.model_path = FLAGS.model_path
        self.logdir = FLAGS.logdir
        self.net_name = FLAGS.net_name
        self.result_path = FLAGS.result_path

