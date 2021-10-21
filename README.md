# IQT
Unofficial implementations of CVPR2021 paper "Perceptual Image Quality Assessment with Transformers" (paper link: https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Cheon_Perceptual_Image_Quality_Assessment_With_Transformers_CVPRW_2021_paper.pdf)

This method accomplishes 1st in the NTIRE2021 Perceptual Image Quality Assessment (PIQA) challenge.

The environmental settings are described below. (I cannot gaurantee if it works on other environments)
- Pytorch=1.7.1 (with cuda 11.0)
- einops=0.3.0
- numpy=1.18.3
- cv2=4.2.0
- scipy=1.4.1
- json=2.0.9
- tqdm=4.45.0

# Train & Validation
First, you need to download weights of InceptionResNetV2 pretrained on ImageNet database.
- Downlod the weights from this website (http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth)
- rename the .pth file as "inceptionresnetv2.pth" and put it in the "model" folder

Second, you need to download the PIPAL database.
- Download the PIPAL database (train/valid) from this website (https://github.com/HaomingCai/PIPAL-dataset)
- set the database path in "train.py" (It is represented as "db_path" in "train.py")
- Please check "PIPAL.txt" is in "IQA_list" folder

After those settings, you can run the train & validation code by running "train.py"
- python3 train.py (execution code)
- This code works on single GPU. If you want to train this code in muti-gpu, you need to change this code
- Options are all included in "train.py". So you should change the variable "config" in "train.py"
![image](https://user-images.githubusercontent.com/77471764/138200352-6d0e1749-3d0d-40b3-a882-98958924a66c.png)

Belows is the performance on PIPAL database
- As we cannot get the ground truth of valid dataset, we separate the train dataset of PIPAL as 8:2
- PLCC: 0.9134

# Inference
First, you need to specify variables in "test.py"
- db_path: root folder of test images
- weight_file: checkpoint file (trained on KonIQ-10k dataset)
- result_file: inference score will be saved on this txt file
![image](https://user-images.githubusercontent.com/77471764/138201052-53df7ca3-d0a7-4cf7-9551-f6a96313bfd4.png)


After those settings, you can run the inference code by running "test.py"
- python3 test.py (execution code)

# Acknolwdgements
We refer to the following website to implement the transformer (https://paul-hyun.github.io/transformer-01/)
