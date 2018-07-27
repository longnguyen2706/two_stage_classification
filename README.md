# Google cloud:

## Install google cloud sdk:
### Create environment variable for correct distribution
```
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"
```

### Add the Cloud SDK distribution URI as a package source
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

### Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

### Update the package list and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk

### Init gcloud 
gcloud init

##Mount 
gcsfuse --implicit-dirs gpu-machine-207012-mlengine /home/ndlong95/gcloud-storage

##Unmount 
fusermount -u /home/ndlong95/gcloud-storage


## Copy file:
### From computer terminal:
gcloud compute scp --recurse  /home/duclong002/pretrained_model/keras/resnet152_weights_tf.h5 k80:~/
gcloud compute scp --recurse /home/duclong002/Dataset/JPEG_data k80:~/
gcloud compute scp --recurse /home/long/Desktop/Hela_split_30_2018-07-19.pickle k80:~/

# Install dependencies
## Keras
```commandline
sudo pip3 install keras==2.1.5
```
## Sklearn
```commandline
pip3 install -U scikit-learn
```
# Run
cd two_stage_classification/
PYTHONPATH='.' python3 finetune/keras_finetune.py 
PYTHONPATH='.' python3 finetune/finetune_master.py \
    --pool_dir  '/home/ndlong95/Hela_split_30_2018-07-19.pickle' \
    --image_dir  '/home/ndlong95/Dataset/JPEG_data/Hela_JPEG' \
    --architecture 'resnet_v2' \
    --start_pool  0 \
    --end_pool 1 \
    --log_dir '/home/ndlong95/finetune/log' \
    --save_model_dir  '/home/ndlong95/finetune/saved_models' \
    --result_dir '/home/ndlong95/finetune/results' \
    --train_batch  8 \
    --test_batch  16

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

# Structure
## handcrafted 
```commandline
Train the SURF
```

## finetune
```commandline
Finetune the CNN
```

## offtheshelf
```commandline
Extract off the shelf features
```

## 2 stage module
### classifier
### trainer
### tester
### generate split data list
### control model log
 
