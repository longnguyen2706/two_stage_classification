import os
# /tmp/bottleneck/Nucleus/r06aug97.gidap.12--1---2.dat.jpg_inception_resnet_v2.txt
import copy
import numpy as np

from split_data import load_pickle, print_split_report


def read_feature_file(filepath):
    with open(filepath, 'r') as feature_file:
        feature_string = feature_file.read()
    try:
        feature_values = [float(x) for x in feature_string.split(',')]
    except ValueError:
        print('Invalid float found')
    return np.asarray(feature_values)

def get_features(list_images, label_names, feature_dir):
    features_list = []
    for i, image_path in enumerate(list_images):
        label_name = label_names[i]
        image_name = image_path.split('/')[-1]
        prefix = os.path.join(feature_dir, label_name, image_name)

        inception_features = read_feature_file(prefix+"_inception_v3.txt")
        resnet_features = read_feature_file(prefix+"_resnet_v2.txt")
        inception_resnet_features = read_feature_file(prefix+"_inception_resnet_v2.txt")
        features= np.concatenate((inception_features, resnet_features, inception_resnet_features))
        assert (features.shape  == ((2048 *2 + 1536),)) # if no exception -> correct
        # print (features.shape)
        features_list.append(features)
    return np.asarray(features_list)

def testing():
    data_pool = load_pickle('/home/long/Desktop/Hela_split_30_2018-07-19.pickle')
    print(data_pool['data']['29']['data_name'])
    print(len(data_pool['data']['29']['train_files']))
    print(data_pool['data']['29']['train_files'])

    split = data_pool['data']['29']
    print_split_report('train', split['train_report'])
    print_split_report('val', split['val_report'])
    print_split_report('test', split['test_report'])
    # print (get_features(split['train_files'], split['train_label_names'], '/mnt/6B7855B538947C4E/Dataset/features/off_the_shelf'))


if __name__ == '__main__':
    testing()

