
from config import FLAGS
import initializer as init
from nets.cloud_matting_net import CloudMattingNet
FLAGS.batch_size=1
params=init.TrainingParamInitialization()

nets=CloudMattingNet(params)
nets.run_test_loop()


