# Two stage classification 
The aims of this project is to develop a 2 stage classification framework with reject option that accurately classify biomedical images. The project contains of several modules, in which a specific type of features are extracted from the images. Because the features are complementary in nature, using each of them for different stage tends to give better result.

In each stage, SVM is used as the classifier. SVM allows easily calculating the confident score of the classification result for each image. If stage 1 reachs high enough confident score, the final result will be stage 1 result. Otherwise, the image will be feed to stage 2 for classification. The two stages need to be trained with the training samples seperately before using in the two stage classification.

The features used are handcrafted features (SURF Bag-of-words), Off-the-shelf CNN features (from Inception-v3, Resnet-v2, Inception-Resnet-v2 that are pretrained on Imagenet without fine-tuning on the biomedical dataset), and the features from finetuning nets (Inception-v3, Resnet-v2, Inception-Resnet-v2 that are finetune on biomedical image dataset). Features from different CNN architectures are also concatenated to have more descriptive representation.

Due to the very large number of deep CNNs to finetune, as well as the computational intensiveness of buiding bag-of-words vocab for handcrafted features, this project introduce a concept named **"data_pool"**. Basically, to evaluate performance of any structure/ framework in general, the experiments need to re-run several times with random train/val/test data split. To make the training of each stage/ type of features distributed, the data split need to remain the same between training for handcrafted/off-the-shelf/finetune. Data pool comes to address this issue. By buiding a list of random split for train/val/test (by default 30 random split), save it, copy it to every machine, the training process of this 2 stage framework can be divide to each machine based on the workload.


In this project, both local machine and google cloud machine (k80-12GB GPU) are used. For faster process, the training can also be divide and run on many google cloud machine.

This project used Keras with Tensorflow backend for finetuning the CNNs; Tensorflow to extract off-the-shelf features; Opencv-python to extract handcrafted features and Sklearn for SVM. 

## Getting started
### Install dependencies 
#### Pip3
```commandline
sudo apt-get install pip3
```
#### Tensorflow (version=1.4-1.8)
```commandline
 See Tensorflow documentation: https://www.tensorflow.org/install/install_linux
```
#### Keras (version=2.1.5)
```commandline
sudo pip3 install keras==2.1.5
```
#### Sklearn
```commandline
pip3 install -U scikit-learn
```
#### Opencv-python3 (tested for unbuntu 16) 
##### This installation include both opencv and opencv-contrib. Opencv contrib is needed for SIFT/SURF extractor
```commandline
pip install opencv-contrib-python
```

### Google cloud machine:
#### Setting for cloud machine:
* NVIDIA k80 (12GB GPU) - **zone us-central1-c**
* 50GB of hard-drive
* Ubuntu 16.04
#### Install google cloud sdk (for Ubuntu local machine):
##### Create environment variable for correct distribution
```
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"
```
##### Add the Cloud SDK distribution URI as a package source
```
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
```
##### Import the Google Cloud Platform public key
```
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
```
##### Update the package list and install the Cloud SDK
```
sudo apt-get update && sudo apt-get install google-cloud-sdk
```
##### Init gcloud 
```commandline
gcloud init
```

#### Transfer file between computer and cloud:
##### Example: From computer to cloud (copy this and paste to computer terminal):
```
gcloud compute scp --recurse  /home/duclong002/pretrained_model/keras/resnet152_weights_tf.h5 ndlong95@k80:~/
gcloud compute scp --recurse /home/duclong002/Dataset/JPEG_data ndlong95@k80:~/
gcloud compute scp --recurse /home/long/Desktop/Hela_split_30_2018-07-19.pickle ndlong95@k80:~/
```
##### Example: From cloud to computer (copy this and paste to computer terminal):
```commandline
gcloud compute scp ndlong95@k80:~/finetune/saved_models/Hela_split_30_2018-07-19_0_resnet_v2.h5 /home/long/Desktop
```

## Run
### Always start from the project root
```commandline
cd two_stage_classification/
```
### First thing first: creating data pool split
* By default, this script will create a .pickle file including train/val/test of 30 independent random split.
* This file will be used across machines to train every stage
```commandline
PYTHONPATH='.' python3 split_data.py
```
### 

```
### Test the finetune only
```commandline
PYTHONPATH='.' python3 finetune/keras_finetune.py 
```

### Run the hyper params tuning
#### Local 
```commandline
PYTHONPATH='.' python3 finetune/finetune_master.py \
    --pool_dir  '/home/long/Desktop/Hela_split_30_2018-07-19.pickle' \
    --image_dir  '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG' \
    --architecture 'inception_resnet_v2' \
    --start_pool  0 \
    --end_pool 1 \
    --log_dir '/home/long/finetune/log' \
    --save_model_dir  '/home/long/finetune/saved_models' \
    --result_dir '/home/long/finetune/results' \
    --train_batch  8 \
    --test_batch  16
```
#### Cloud
```
PYTHONPATH='.' python3 finetune/finetune_master.py \
    --pool_dir  '/home/long/Desktop/Hela_split_30_2018-07-19.pickle' \
    --image_dir  '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG' \
    --architecture 'resnet_v2' \
    --start_pool  0 \
    --end_pool 0 \
    --log_dir '/home/long/finetune/log' \
    --save_model_dir  '/home/long/finetune/saved_models' \
    --result_dir '/home/long/finetune/results' \
    --train_batch  8 \
    --test_batch  16

```

## Structure
### handcrafted 
```commandline
Train the SURF
```

### finetune
```commandline
Finetune the CNN
```

### offtheshelf
```commandline
Extract off the shelf features
```

### 2 stage module
#### classifier
#### trainer
#### tester
#### generate split data list
#### control model log
 