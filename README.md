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
### Finetuning 
#### Test the finetune only
```commandline
PYTHONPATH='.' python3 finetune/keras_finetune.py 
```

#### Run the hyper params tuning
* The printout (log) during the training will be dump to file (>> means appends to file)
* architecture: may be one of the following: 'inception_v3', 'inception_resnet_v2', 'resnet_v2'
* start_pool, end_pool: the hyper params tuning will run for pool that have index from (start_pool to end_pool). For ex, start_pool = 0, end_pool=1 means pool 0 and 1 will be trained, and the resutl will be recorded to a single file. By default, there are 30 pools from pool 0 to pool 29.
##### On Cloud
```commandline
PYTHONPATH='.' python3 finetune/finetune_master.py \
    --pool_dir  '/home/ndlong95/Hela_split_30_2018-07-19.pickle' \
    --image_dir  '/home/ndlong95/Dataset/JPEG_data/Hela_JPEG' \
    --architecture 'inception_resnet_v2' \
    --start_pool  0 \
    --end_pool 1 \
    --log_dir '/home/ndlong95/finetune/log' \
    --save_model_dir  '/home/ndlong95/finetune/saved_models' \
    --result_dir '/home/ndlong95/finetune/results' \
    --train_batch  16 \
    --test_batch  32 >>/home/ndlong95/finetune_log.txt
```

##### On Local 
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
    --test_batch  16 >>/home/long/finetune_log.txt

```
### Extract handcrafted features/off_the_shelf features:
* You can just run the file if you are using pycharm. The file is not complete yet but the main functions are working.

## Project Structure
* **split_data.py:** create .pickle file that contains random train/val/test data split. By default, the file contains 30 random split.
* **svm_classifier.py:** contains SVM_CLASSIFIER class which has train/test/save/load method for SVM model. Also, this class provide method to calculate confidence score of classification result, which is applicable to SVM linear classifier 
* **main.py (deprecated):** should load CNN/finetune/handcrafted features, train SVM of each stage, train reject option and produce the final result
### Finetune/
* The finetune package is written with Keras 2. The structure is as follow:
    * **keras_finetune.py** 
        * contains necessary function to finetune a single net structure with single params setting, for a single data pool split
        * the train() function return (acc, loss) score for train/val/test that will be used for hyper params optimization
        * also contain restore trained model function, but got issue (see known issue) 
    * **finetune_master.py**
        * do hyper params optimization by repeatly call train() in keras_finetune.py with different params settings. The best params is chosen based on validation acc
        * the script also take in different setting such as batch_size, log_dir ... to save the best model weights and tensorboard log to file. Also, it will save all the result to .pickle file
    * **extract_finetune_features.py (uncompleted)**
        * load the trained weights, freeze the model graph 
        * take the data split, feed to the input layers and take the features before softmax layer. record to file
        * dump features to .pickle
    * **misc/**
        * **data_generator.py:** return train/val/test generator needed by model.fit_generator during training
        * **utils.py:** contains necessary functions for this finetune module
    * **net/**
        * **resnet152.py**: declare resnet152 model. keras.applications does not have resnet152 (only resnet50). **This resnet also come with the weight that need to be specified.** Please see create_model_info() in keras_finetune.py
        * **custom_layers.py**: custom layers needed for resnet152 model
    * **model/**
        * should contain the model so that resnet152 model can be restored from the finetune weight
   
### off_the_shelf/
* The extracted features (.txt file) need to be copied to the machine. This is the features obtained from the last paper
* **extract_CNN_features.py:** read features from the file, return concatenation features

### handcrafted/
* **bow/**
    * **surf_bow.py:** contains SURF_BOW class that build extract SURF features, build vocab (BOW) from train images and return histogram of each train/val/test images that will be used in classification
* **extract_bow_features.py:** extract features for pools

## Todo/Known issues
### Known issues:
* finetune/misc/data_generators.py cannot shuffle train/val/test batch: it is due to (data, label) are not correctly shuffled. However, without shuffling (passing shuffle=False), the model still perform very well
* finetune/keras_finetune.py: the restore_model to restore trained model works well with 'inception_v3', 'inception_resnet_v2' architecture but cannot load 'resnet_v2' due to the custom layer in resnet model. It can be fixed by having resnet model declaration and just load the weight .h5 file instead of loading both .json and .h5 file

### TODO:
* extract features from finetune model
* modify extract_CNN_features.py to extract features from pool 
* modify main.py to work with pool, load features from cnn/handcrafted/finetune
* write function to train threshold t with respect to rejection rate alpha
