'''
Cloud Mating GAN
Description: Parameters Configuration
Author: Wenyuan Li
Date: Feb., 2019
'''

from config import FLAGS
import initializer as init
from nets.cloud_matting_net import CloudMattingNet

FLAGS.batch_size = 1
params = init.TrainingParamInitialization()

nets = CloudMattingNet(params)
nets.run_test_loop()

