from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import copy
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.utils import Bunch
import collections
import load_CNN_features
import seaborn as sns

from sift.sift_bow import SIFT_BOW
from surf.surf_bow import SURF_BOW
from svm_classifier import SVM_CLASSIFIER

sns.set()

IMAGE_DIR = '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG'
FEATURE_DIR = '/mnt/6B7855B538947C4E/Dataset/features/off_the_shelf'
OUT_MODEL1 = '/mnt/6B7855B538947C4E/home/duclong002/handcraft_models/stage1.pkl'
OUT_MODEL2 = '/mnt/6B7855B538947C4E/handcraft_models/stage2.pkl'
# PARAM_GRID = {'linearsvc__C': [1, 5, 10, 50]}

HYPER_PARAMS_1 = [
    {
        'pow_min': -15,
        'pow_max': 15,
        'base': 2,
        'pow_step': 1,
        'type': 'linearsvc__C',
    },
]
HYPER_PARAMS_2 = [
    {
        'pow_min': -15,
        'pow_max': 15,
        'base': 2,
        'pow_step': 1,
        'type': 'svc__C',
    },
    {
        'pow_min': -5,
        'pow_max': 5,
        'base': 2,
        'pow_step': 1,
        'type': 'svc__gamma'
    }
]

CLASSIFIER_1 = svm.LinearSVC()
CLASSIFIER_2 = svm.SVC(kernel='rbf', class_weight='balanced')
DIM_REDUCER = PCA(n_components=300, whiten=True, random_state=42,svd_solver='randomized')
NUM_OF_WORDS = 1000
T = [0.35, 0.40, 0.45,  0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


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
                files_and_labels.append((os.path.join(self.directory, label, f), label))

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
        print(dataset.data.shape)
        # print(dataset.label_names)
        train_files, test_files, train_labels, test_labels, train_label_names, test_label_names \
            = train_test_split(dataset.data, dataset.labels, dataset.label_names, test_size=self.test_size)
        train_files, val_files, train_labels, val_labels, train_label_names, val_label_names \
            = train_test_split(train_files, train_labels, train_label_names, test_size=self.val_size)

        print('train size: ', train_labels.shape)
        self.data_split_report(train_label_names, 'train')
        self.data_split_report(val_label_names,'val' )
        self.data_split_report(test_label_names, 'test')

        return train_files, train_labels, train_label_names, \
               val_files, val_labels, val_label_names, \
               test_files, test_labels, test_label_names, self.class_names

    def data_split_report(self, label_names, set_name):
        class_freq = collections.Counter(label_names)
        print ("class freq for set %s "% set_name)
        print('*********')
        for key in sorted(class_freq):
            print( "%s: %s" % (key, class_freq[key]))
        print("-----------------------------------")

def gen_grid(hyper_params):
    params_grid ={}
    for hyper_param in hyper_params:
        grid_params = []
        for i in range(hyper_param['pow_max'] - hyper_param['pow_min'] + 1):
            if (i % hyper_param['pow_step'] == 0):
                grid_params.append(pow(hyper_param['base'], hyper_param['pow_min'] + i))
        params_grid[str(hyper_param['type'])]=grid_params
    print('param grids for HYPER PARAMS: ', hyper_params, params_grid)
    return params_grid


def get_CNN_features(train_files, train_labels, train_label_names,
                     val_files, val_labels, val_label_names,
                     test_files, test_labels, test_label_names):
    train_CNN_features = load_CNN_features.get_features(train_files, train_label_names, FEATURE_DIR)
    val_CNN_features = load_CNN_features.get_features(val_files, val_label_names, FEATURE_DIR)
    test_CNN_features = load_CNN_features.get_features(test_files, test_label_names, FEATURE_DIR)
    return train_CNN_features, val_CNN_features, test_CNN_features


def get_BOW_features(train_files, train_labels, train_label_names,
                     val_files, val_labels, val_label_names,
                     test_files, test_labels, test_label_names):
    surf_bow = SURF_BOW(num_of_words=NUM_OF_WORDS)
    surf_bow.build_vocab(train_files)
    train_surf_features = surf_bow.extract_bow_hists(train_files)
    val_surf_features = surf_bow.extract_bow_hists(val_files)
    test_surf_features = surf_bow.extract_bow_hists(test_files)
    return train_surf_features, val_surf_features, test_surf_features


def find_best_t(cls1, cls2, dataset, CNN_features, surf_features, labels, class_names):
    accuracies = []
    for t in T:
       result= get_2_stage_performance(cls1, cls2, dataset, CNN_features, surf_features, labels, class_names, t)
       acc = result['accuracy']
       accuracies.append(acc)
    best_acc =  max(accuracies)
    best_t = T[np.argmax(accuracies)]
    return best_t, best_acc # TODO: return best recall

def get_2_stage_performance(cls1, cls2, dataset, CNN_features, surf_features, labels, class_names, t):
    Y = []
    for i, features in enumerate(CNN_features):
        y1 = cls1.trained_model.predict([features])[0]
        cs = cls1.cal_CS(features, y1, dataset.categories)
        if (cs < 1 - t):
            # print("*** Stage 1 reject with t, cs = ", t, cs, " ***")
            features_bow = surf_features[i]
            y2 = cls2.trained_model.predict([features_bow])[0]
            # print("*** y1, y2: ", y1, y2, " ***")
            Y.append(y2)
        else:
            # print("*** Stage 1 accept with t, cs = ", t, cs, " ***")
            Y.append(y1)
    print("Classification report with t = ", t)
    print(classification_report(labels,Y,
                                target_names=class_names))
    print("----------------------------")

    # now call precision
    precision, recall, fscore, support = score(labels, Y)
    accuracy = accuracy_score(labels, Y)
    print('accuracy: ', accuracy)
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))

    average_precision = 0
    for p in precision:
        average_precision = average_precision + p / len(precision)
    # print('average precision: ', average_precision)
    return {'accuracy': accuracy, 'average_precision': average_precision, 'precision': precision, 'recall': recall, 'fscore': fscore,
            'support': support}

def cal_mean_and_std(result_arr, name):
    mean = sum(result_arr) / float(len(result_arr))
    std = np.std(result_arr, dtype=np.float32, ddof=1)
    print("average  %s result" % str(name), mean)
    print("standard dev of %s" % str(name), std)
    print ("_________________________________________")
    return mean, std


def main():
    all_acc_val_CNN = [] # all accuracy CNN
    all_acc_val_BOW = []
    all_acc_val_2_stage = []
    all_acc_test_CNN = []
    all_acc_test_BOW = []
    all_acc_test_2_stage = []

    for i in range (30):
        print ("Train model ith = %s/" % str(i+1), str(30))
        dataset = MyDataset(directory=IMAGE_DIR, test_size=0.2, val_size=0.25) #0.2 0.25
        train_files, train_labels, train_label_names, \
        val_files, val_labels, val_label_names, \
        test_files, test_labels, test_label_names, class_names = dataset.get_data()

        params_grid_1 = gen_grid(HYPER_PARAMS_1)
        params_grid_2 = gen_grid(HYPER_PARAMS_2)

        train_CNN_features, val_CNN_features, test_CNN_features = get_CNN_features(
            train_files, train_labels, train_label_names,
            val_files, val_labels, val_label_names,
            test_files, test_labels, test_label_names)

        train_surf_features, val_surf_features, test_surf_features = get_BOW_features(
            train_files, train_labels, train_label_names,
            val_files, val_labels, val_label_names,
            test_files, test_labels, test_label_names
        )

        # now train stage 1
        cls1 = SVM_CLASSIFIER(params_grid_1, CLASSIFIER_1, OUT_MODEL1)
        cls1.prepare_model()
        cls1.train(train_CNN_features, train_labels)
        print("Finish train stage 1")

        print("Now eval stage 1 on val set")
        cls1_val = cls1.test(val_CNN_features, val_labels,class_names)
        acc_val_CNN =cls1_val['accuracy']
        all_acc_val_CNN.append(acc_val_CNN)

        print("Now eval stage 1 on test set")
        cls1_test= cls1.test(test_CNN_features, test_labels, class_names)
        acc_test_CNN = cls1_test['accuracy']
        all_acc_test_CNN.append(acc_test_CNN)
        print("---------------------")

        # now train stage 2
        cls2 = SVM_CLASSIFIER(params_grid_2, CLASSIFIER_2, OUT_MODEL2)
        cls2.prepare_model()
        cls2.train(train_surf_features, train_labels)
        print("Finish train stage 2")

        print("Now eval stage 2 on val set")
        cls2_val = cls2.test(val_surf_features, val_labels, class_names)
        acc_val_BOW = cls2_val['accuracy']
        all_acc_val_BOW.append(acc_val_BOW)

        print("Now eval stage 2 on test set")
        cls2_test = cls2.test(test_surf_features, test_labels, class_names)
        acc_test_BOW = cls2_test['accuracy']
        all_acc_test_BOW.append(acc_test_BOW)
        print("---------------------")

        # now train rejection rate
        cls1.get_centroids(train_CNN_features, train_labels, dataset.categories)
        print("Now eval 2 stages on val set: ")
        t, acc_val_2_stage = find_best_t(cls1, cls2, dataset, val_CNN_features, val_surf_features, val_labels, class_names)
        print ('The best t, val acc is ', t, acc_val_2_stage)
        all_acc_val_2_stage.append(acc_val_2_stage)

        print("Now eval 2 stages on test set: ")
        test_2_stage =  get_2_stage_performance(cls1, cls2, dataset, test_CNN_features,
                                                    test_surf_features, test_labels, class_names, t)
        acc_test_2_stage = test_2_stage['accuracy']
        all_acc_test_2_stage.append(acc_test_2_stage)


    cal_mean_and_std(all_acc_val_CNN, "val_CNN")
    cal_mean_and_std(all_acc_val_BOW, "val_BOW")
    cal_mean_and_std(all_acc_val_2_stage, "val_2_stage")
    cal_mean_and_std(all_acc_test_CNN, "test_CNN")
    cal_mean_and_std(all_acc_test_BOW, "test_BOW")
    cal_mean_and_std(all_acc_test_2_stage, "test_2_stage")


if __name__ == '__main__':
    main()
