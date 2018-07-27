from __future__ import absolute_import

import collections
import copy

from keras import Model, optimizers
from keras.applications import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import TensorBoard, EarlyStopping
from keras.initializers import TruncatedNormal
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, AveragePooling2D
from keras.models import load_model
from finetune.misc.data_generator import DataGenerator
from sklearn.cross_validation import train_test_split
from sklearn.utils import Bunch

from misc.data_generator import get_generators
from misc.utils import *
from net.resnet152 import resnet152_model

def resnet_model(model_info, num_classes):
    base_model = resnet152_model(model_info['input_width'], model_info['input_height'], model_info['input_depth'],
                                 model_info['pretrained_weights'], num_classes)


    num_base_layers = len(base_model.layers)
    init = TruncatedNormal(mean=0.0, stddev=0.001, seed=None)
    input = base_model.input
    x = base_model.output
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8', kernel_initializer=init)(x_newfc)
    model = Model(input=input, outputs=x_newfc)

    return model, num_base_layers
