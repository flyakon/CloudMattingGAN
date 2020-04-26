"""
Cloud Generation
Description: main
Author: Zhengxia Zou
Date: Feb., 2019
"""


import initializer as init
from nets.cloud_gan import CloudGAN
# All parameters used in this file

if __name__=='__main__':
    params=init.TrainingParamInitialization()
    nets=CloudGAN(params)
    nets.run_train_loop()