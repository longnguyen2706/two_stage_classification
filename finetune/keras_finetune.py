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

GENERAL_SETTING = {
    'early_stopping_n_steps': 5,
    'batch_size': 1,
    'eval_step_interval': 100,
    'flip_left_right': False,
    'output_labels': '/tmp/output_labels.txt',
    'random_brightness': 0,
    'random_crop': 0,
    'random_scale': 0,
    'test_batch_size': -1,
    'testing_percentage': 20,
    'validation_percentage': 10,
    'validation_batch_size': -1,
    # 'csvlogfile': csv_log_directory,
    'how_many_training_steps': 10000,
    'image_dir': '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG/',
    # 'summaries_dir': summaries_directory
}
log_dir = '/mnt/6B7855B538947C4E/logdir/keras/'

def create_model_info(architecture):
    model_info = {}
    if architecture == 'inception_v3':
        model_info['bottleneck_tensor_size'] = 2048
        model_info['input_width'] = 299
        model_info['input_height'] = 299
        model_info['input_depth'] = 3
        model_info['input_mean'] = 128
        model_info['input_std'] = 128
        model_info['pretrained_weights'] = None

    elif architecture == 'resnet_v2':
        model_info['bottleneck_tensor_size'] = 2048
        model_info['input_width'] = 224
        model_info['input_height'] = 224
        model_info['input_depth'] = 3
        model_info['input_mean'] = 128
        model_info['input_std'] = 128
        model_info['pretrained_weights'] = '/mnt/6B7855B538947C4E/pretrained_model/keras/resnet152_weights_tf.h5'

    elif architecture == 'inception_resnet_v2':
        model_info['bottleneck_tensor_size'] = 1536
        model_info['input_width'] = 299
        model_info['input_height'] = 299
        model_info['input_depth'] = 3
        model_info['input_mean'] = 128
        model_info['input_std'] = 128
        model_info['pretrained_weights'] = None

    return model_info

def get_image_lists(image_dir, testing_percentage, validation_percentage):
    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(image_dir, testing_percentage, validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + image_dir)
        return -1
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         image_dir +
                         ' - multiple classes are needed for classification.')
        return -1

    return image_lists, class_count

#
# def get_generators(image_lists, model_info):
#     sess = tf.Session()
#     with sess.as_default():
#         # Set up the image decoding sub-graph.
#         jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
#             model_info['input_width'], model_info['input_height'],
#             model_info['input_depth'], model_info['input_mean'],
#             model_info['input_std'])
#
#         # training_generator = DataGenerator(partition['train'], labels, **params)
#         # validation_generator = DataGenerator(partition['validation'], labels, **params)
#         train_generator = DataGenerator().generate(sess, image_lists, GENERAL_SETTING['batch_size'], 'training',
#                                                    GENERAL_SETTING['image_dir'], jpeg_data_tensor, decoded_image_tensor)
#         validation_generator = DataGenerator().generate(sess, image_lists, GENERAL_SETTING['batch_size'], 'validation',
#                                                         GENERAL_SETTING['image_dir'], jpeg_data_tensor,
#                                                         decoded_image_tensor) #TODO: remove this general setting
#
#         test_generator = DataGenerator().generate(sess, image_lists, GENERAL_SETTING['batch_size'], 'testing',
#                                                   GENERAL_SETTING['image_dir'], jpeg_data_tensor, decoded_image_tensor)
#     return train_generator, validation_generator, test_generator


def get_model(num_classes, architecture, model_info, dropout=0,  weights='imagenet'):
    if architecture == 'inception_v3':
        base_model = InceptionV3(weights = weights, include_top=False)

    elif architecture == 'inception_resnet_v2':
        base_model = InceptionResNetV2(weights=weights, include_top=False)

    elif architecture == 'resnet_v2':
        base_model = resnet152_model(model_info['input_width'], model_info['input_height'], model_info['input_depth'],
                                     model_info['pretrained_weights'], num_classes)

    num_base_layers = len(base_model.layers)
    init = TruncatedNormal(mean=0.0, stddev=0.001, seed=None)
    input = base_model.input
    x = base_model.output

    if architecture == 'inception_v3' or 'inception_resnet_v2':
        init = TruncatedNormal(mean=0.0, stddev=0.001, seed=None)
        input = base_model.input
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # x = Dense(64, input_shape=(2048,), activation='relu',  kernel_initializer=init)(x)
        x = Dropout(dropout)(x)

        predictions = Dense(num_classes, input_shape=(model_info['bottleneck_tensor_size'],), activation='softmax', kernel_initializer=init)(x)
        model = Model(input=input, outputs=predictions)

        return model, num_base_layers

    elif architecture == 'resnet_v2':
        x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
        x_newfc = Flatten()(x_newfc)
        x_newfc = Dropout(dropout)(x_newfc)
        x_newfc = Dense(num_classes, activation='softmax', name='fc8', kernel_initializer=init)(x_newfc)
        model = Model(input=input, outputs=x_newfc)

        return model, num_base_layers


def set_model_trainable(model, num_base_layers, layer_to_begin_finetune = -1):
    if layer_to_begin_finetune == -1: # retrain all layers
        for layer in model.layers[:num_base_layers]:
            layer.trainable = True

    elif layer_to_begin_finetune <= num_base_layers:
        for layer in model.layers[:layer_to_begin_finetune]:
            layer.trainable = False
        for layer in model.layers[layer_to_begin_finetune:]:
            layer.trainable = True

    print(model.summary())

    return model

def train(image_dir, testing_percentage, validation_percentage, batch_size, architecture):
    tf.logging.set_verbosity(tf.logging.INFO)

    model_info = create_model_info(architecture)

    # get data
    image_lists, num_classes = get_image_lists(image_dir, testing_percentage, validation_percentage)
    # train_generator, validation_generator, test_generator = get_generators(image_lists, model_info)
    train_generator, validation_generator, test_generator = get_generators()
    # get model
    model, num_base_layers = get_model(num_classes, architecture, model_info)

    model = set_model_trainable(model, num_base_layers)

    # optimizer = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)  # Inception
    optimizer = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)  # Inception-Resnet
    # optimizer = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.99)
    # optimizer = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=GENERAL_SETTING['batch_size'],
                              write_graph=True, write_grads=False)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0,
                                   mode='auto')

    train_len = int(842 * 0.6)
    validation_len = int(842 * 0.2) # TODO: fix this
    test_len = int(842 * 0.2)

    '''
    Train the model 
    '''
    # note that keras 2 have problems with sample_per_epochs -> need to use sample per epoch
    # see https://github.com/keras-team/keras/issues/5818
    model.fit_generator(
        train_generator,
        epochs=100,
        steps_per_epoch=train_len // 8+1,
        validation_data=validation_generator,
        validation_steps=validation_len // 16+1,
        callbacks=[tensorboard, early_stopping],
    )

    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name)
    #
    # # we chose to train the top 2 inception blocks, i.e. we will freeze
    # # the first 249 layers and unfreeze the rest:
    # # for layer in model.layers[:249]:
    # #     layer.trainable = False
    # # for layer in model.layers[249:]:
    # #     layer.trainable = True
    #
    # # we need to recompile the model for these modifications to take effect
    # # we use SGD with a low learning rate
    # from keras.optimizers import SGD
    # optimizer = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.99)
    # # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
    #
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit_generator(
    #     train_generator,
    #     epochs=100,
    #     steps_per_epoch=train_len // GENERAL_SETTING['batch_size'],
    #     validation_data=validation_generator,
    #     validation_steps=validation_len // GENERAL_SETTING['batch_size'],
    #     callbacks=[tensorboard]
    # )
    #
    score = model.evaluate_generator(test_generator, test_len)
    print("score", score)

    score = model.evaluate_generator(test_generator, test_len)
    print("score", score)

    score = model.evaluate_generator(test_generator, test_len)
    print("score", score)

    score = model.evaluate_generator(test_generator, test_len*2)
    print("score", score)

    score = model.evaluate_generator(test_generator, test_len*5)
    print("score", score)

    model.save('/home/long/keras_resnet.h5')

def restore_model(model_path):
    model = load_model(model_path)
    print('Restored model from path ', model_path)
    print (model.summary())


def main(_):
    train(GENERAL_SETTING['image_dir'], 20, 10, 8, 'inception_v3')
    # image_lists = get_image_lists(GENERAL_SETTING['image_dir'], 20, 10)
    # print(image_lists)
if __name__ == '__main__':
      tf.app.run(main=main)
