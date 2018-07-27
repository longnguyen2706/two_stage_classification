import collections
import copy
import datetime
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from utils import current_date, dump_pickle, load_pickle


class MyDataset():
    def __init__(self, directory, test_size, val_size):
        self.directory = directory
        self.filenames = None
        self.labels = None
        self.label_names = None
        self.class_names = None
        self.categories = None
        self.test_size = test_size
        self.val_size = val_size

    def list_images(self):
        self.labels = os.listdir(self.directory)
        self.labels.sort()

        files_and_labels = []
        for label in self.labels:
            for f in os.listdir(os.path.join(self.directory, label)):
                # files_and_labels.append((os.path.join(self.directory, label, f), label)) # full path to image
                files_and_labels.append((os.path.join(label, f), label)) # only dir/imagename

        self.filenames, self.labels = zip(*files_and_labels)
        self.filenames = list(self.filenames)
        self.labels = list(self.labels)
        self.label_names = copy.copy(self.labels)
        unique_labels = list(set(self.labels))
        unique_labels.sort()

        label_to_int = {}
        for i, label in enumerate(unique_labels):
            label_to_int[label] = i

        self.labels = [label_to_int[l] for l in self.labels]
        self.class_names = unique_labels
        self.categories = list(set(self.labels))
        return

    def get_data(self):
        self.list_images()  # get image list

        dataset = Bunch(
            data=np.asarray(self.filenames),
            label_names=np.asarray(self.label_names), labels=np.asarray(self.labels),
            DESCR="Dataset"
        )
        print('dataset size: ', dataset.data.shape)
        # print(dataset.label_names)
        train_files, test_files, train_labels, test_labels, train_label_names, test_label_names \
            = train_test_split(dataset.data, dataset.labels, dataset.label_names, test_size=self.test_size)
        train_files, val_files, train_labels, val_labels, train_label_names, val_label_names \
            = train_test_split(train_files, train_labels, train_label_names, test_size=self.val_size)

        print('train size: ', train_labels.shape)

        return train_files, train_labels, train_label_names, \
               val_files, val_labels, val_label_names, \
               test_files, test_labels, test_label_names, self.class_names

    def data_split_report(self, label_names, set_name):
        class_freq = collections.Counter(label_names)
        print_split_report(set_name, class_freq)
        return class_freq


def print_split_report(set_name, class_freq):
    print ("class freq for set %s "% set_name)
    print('*********')
    for key in sorted(class_freq):
        print( "%s: %s" % (key, class_freq[key]))
    print("-----------------------------------")
    return

'''
    Since dict is unordered -> need to 
'''
def gen_data_pool(dataset_name, dataset_dir, path, test_size=0.2, val_size=0.25, pool_size=30):
    now = datetime.datetime.now()
    date = current_date(now)
    pool = {}
    pool_name = dataset_name+'_split_'+str(pool_size)+'_'+str(date)
    pool['pool_name'] = pool_name
    pool['data'] = {}

    for i in range (pool_size):
        print ("Generate dataset split %sth"% str(i+1))
        dataset = MyDataset(dataset_dir, test_size, val_size)

        train_files, train_labels, train_label_names, \
        val_files, val_labels, val_label_names, \
        test_files, test_labels, test_label_names, class_names = dataset.get_data()

        train_report = dataset.data_split_report(train_label_names, 'train')
        val_report= dataset.data_split_report(val_label_names, 'val')
        test_report = dataset.data_split_report(test_label_names, 'test')

        data_i = {}
        data_i['data_name'] = dataset_name+'_'+str(i) +'_' + date
        data_i['train_files'] = train_files
        data_i['train_labels'] = train_labels
        data_i['train_label_names'] = train_label_names
        data_i['train_report'] = train_report

        data_i['test_files'] = test_files
        data_i['test_labels'] = test_labels
        data_i['test_label_names'] = test_label_names
        data_i['test_report'] = test_report

        data_i['val_files'] = val_files
        data_i['val_labels'] = val_labels
        data_i['val_label_names'] = val_label_names
        data_i['val_report'] = val_report

        data_i['class_names'] = class_names

        pool['data'][str(i)]=data_i
        print ('Appended split %sth to pool' %str(i+1))
        print('____________________________________')

    # dump to file
    path = os.path.join(path, pool_name)
    filepath = dump_pickle(pool, path)
    return pool, filepath


def main():
    # need to change dir to your appropriate dir
    pool, filepath = gen_data_pool('Hela', '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG', '/home/long/Desktop/')
    print (filepath)

    # test the result
    dict = load_pickle(filepath)
    # print (dict)
    split_1= dict['data']['0']
    train_report = split_1['train_report']
    print_split_report('train', train_report)

if __name__ == '__main__':
    main()












