import os
# /tmp/bottleneck/Nucleus/r06aug97.gidap.12--1---2.dat.jpg_inception_resnet_v2.txt
import copy
import numpy as np
from sklearn.utils import Bunch


def read_feature_file(filepath):
    with open(filepath, 'r') as feature_file:
        feature_string = feature_file.read()
    try:
        feature_values = [float(x) for x in feature_string.split(',')]
    except ValueError:
        print('Invalid float found')
    return np.asarray(feature_values)

# TODO: can remove this
# extract list of unique prefix:
# ex: extract /tmp/bottleneck/Nucleus/r06aug97.gidap.12--1---2.dat
# from: /tmp/bottleneck/Nucleus/r06aug97.gidap.12--1---2.dat.jpg_inception_resnet_v2.txt
# /tmp/bottleneck/Nucleus/r06aug97.gidap.12--1---2.dat.jpg_resnet_v2.txt
def get_unique_feature_prefix(directory):
    labels = os.listdir(directory)
    labels.sort()
    file_list = []

    for label in labels:
        for file_name in os.listdir(os.path.join(directory, label)):
            file_prefix= file_name.split('.jpg_')[0]
            file_list.append(os.path.join(directory, label, file_prefix))
    unique_feature_prefix_list = list(set(file_list))
    return unique_feature_prefix_list

# TODO: can remove
def get_labels_and_features(prefix_list):
    label_name_list = []
    features_list = []

    for prefix in prefix_list:
        label_name = prefix.split('/')[-2]
        label_name_list.append(label_name)

        inception_features = read_feature_file(prefix+".jpg_inception_v3.txt")
        resnet_features = read_feature_file(prefix+".jpg_resnet_v2.txt")
        inception_resnet_features = read_feature_file(prefix+".jpg_inception_resnet_v2.txt")
        features= np.concatenate((inception_features, resnet_features, inception_resnet_features))
        assert (features.shape  == ((2048 *2 + 1536),)) # if no exception -> correct
        # print (features.shape)
        features_list.append(features)

    unique_labels = list(set(label_name_list))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i
    label_list  = copy.copy(label_name_list)
    label_list = [label_to_int[l] for l in label_list]

    return label_name_list, label_list, features_list


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

# TODO: can remove
def get_dataset(directory):
    unique_feature_prefix_list = get_unique_feature_prefix(directory)
    label_name_list, label_list, features_list = get_labels_and_features(unique_feature_prefix_list)
    return Bunch(
        data=np.asarray(features_list),
        label_names=np.asarray(label_name_list), labels=np.asarray(label_list),
        DESCR="Dataset"
    )

def main():
    dataset = get_dataset('/mnt/6B7855B538947C4E/Dataset/features/off_the_shelf')
    print (dataset.data.shape)

if __name__ == '__main__':
    main()


