# Overview

Generative Adversarial Training for Weakly Supervised Cloud Matting, in ICCV 2019. [Paper](<http://openaccess.thecvf.com/content_ICCV_2019/papers/Zou_Generative_Adversarial_Training_for_Weakly_Supervised_Cloud_Matting_ICCV_2019_paper.pdf>)

We re-examine the cloud detection under a totally different point of view, i.e. to formulate it as a mixed energy separation process between foreground and background images, which can be equivalently implemented under an image matting paradigm with a clear physical significance. We further propose a generative adversarial framework where the training of our model neither requires any pixel-wise ground truth reference nor any additional user interactions. Our model consists of three networks, a cloud generator G, a cloud discriminator D, and a cloud matting network F, where G and D aim to generate realistic and physically meaningful cloud images by adversarial training, and F learns to predict the cloud reflectance and attenuation. 

![Overview](fig/overview.png)

# Requirements

- Python 3.5
- Tensorflow 1.9
- CUDA 9.0
- CUDNN 7.5

See also in [Requirements.txt](requirements.txt).

# Setup

1. Clone this repo:

   ```
   $CODE_PATH=path_to_code
   git clone ssss $CODE_PATH
   cd $CODE_PATH
   ```

2. Training Cloud Matting Net

   It is recommended firstly to train the Generative Adversarial Networks for 2000 steps.(Optional)

   ``````python
   python cloud_generation.py --checkpoint_gan=model/LSGAN  \
   						   --gan_model=LSGAN \
   						   --sample_dir=sample \
   						   --iter_step=2001 \
   						   --image_dir=dataset/train
   ``````



   ``````python
   python train.py --gan_model=LSGAN \
   				--batch_size=2 \
   				--model_path=model/CloudMattingNet_LSGAN \
   				--net_name=mattingnet \
   				--logdir=log \
   				--sample_dir=sample \
   				--iter_step=38002 \
   				--image_dir=dataset/train \
   				--checkpoint_gan=model/LSGAN \
   				--sample_dir=sample
   ``````


4. Test Cloud Matting Net

   ``````
   python test.py  --model_path=model/CloudMattingNet_LSGAN \
   				--net_name=mattingnet  \
   				--testDataPath=dataset\test\thin_slice \
   				--result_path=result
   ``````

