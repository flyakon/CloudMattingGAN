'''
Cloud Mating GAN
Description: Program entry
Author: Wenyuan Li
Date: Feb., 2019
'''

from config import FLAGS
import initializer as init
from nets.cloud_matting_net import CloudMattingNet

params = init.TrainingParamInitialization()
nets = CloudMattingNet(params)
nets.run_train_loop()

