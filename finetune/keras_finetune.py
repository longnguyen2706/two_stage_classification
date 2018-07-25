from __future__ import absolute_import

from keras import Model, optimizers, metrics
from keras.applications import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import TensorBoard, EarlyStopping
from keras.initializers import TruncatedNormal
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, AveragePooling2D
from keras.models import load_model
from keras import backend as K
import tensorflow as tf

from misc.data_generator import get_generators
from misc.utils import *
from net.resnet152 import resnet152_model

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

def declare_model(num_classes, architecture, model_info, dropout=0, weights='imagenet'):
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

def set_model_trainable(model, num_base_layers, num_of_last_layer_finetune):
    if num_of_last_layer_finetune == -1: # retrain all layers
        for layer in model.layers[:num_base_layers]:
            layer.trainable = True

    elif num_of_last_layer_finetune <= num_base_layers:
        for layer in model.layers[:(num_base_layers-num_of_last_layer_finetune)]:
            layer.trainable = False
        for layer in model.layers[(num_base_layers-num_of_last_layer_finetune):]:
            layer.trainable = True

    print(model.summary())

    return model

#TODO: save train log, return performance result
#TODO: retrain some layers with small learning rate after finetuning -> can do it later by restoring and train few last layers
#TODO: export to pb file
# return val_score, test_score in dict form: test_score = {'acc': model accuracy, 'loss', model loss}
def train(split, image_dir, architecture, hyper_params, log_path = None, save_model_path = None, restore_model_path = None,
          train_batch=8, test_batch=16, num_last_layer_to_finetune=-1):
    tf.logging.set_verbosity(tf.logging.INFO)

    model_info = create_model_info(architecture)

    train_generator, validation_generator, test_generator = get_generators(model_info, split, image_dir, train_batch, test_batch)

    num_classes = len(split['class_names'])
    train_len = len(split['train_files'])
    validation_len = len(split['val_files'])
    test_len = len(split['test_files'])

    # train the model from scratch or train the model from some point
    if restore_model_path == None:
        model, num_base_layers = declare_model(num_classes, architecture, model_info)
        model = set_model_trainable(model, num_base_layers, num_last_layer_to_finetune)
    else:
        model, num_layers = restore_model(restore_model_path)
        model = set_model_trainable(model, num_layers, num_last_layer_to_finetune)

    print ('training the model with hyper params: ', hyper_params)
    optimizer = optimizers.SGD(lr=hyper_params['lr'], decay=hyper_params['lr_decay'],
                               momentum=hyper_params['momentum'], nesterov=hyper_params['nesretov'])  # Inception
    # optimizer = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)  # Inception-Resnet
    # optimizer = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.99)
    # optimizer = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics={'acc': 'accuracy','loss': 'crossentropy'}) # cal accuracy and loss of the model; result will be a dict

    tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0, batch_size=train_batch,
                              write_graph=True, write_grads=False)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0,
                                   mode='auto')


    '''
    Train the model 
    '''
    # note that keras 2 have problems with sample_per_epochs -> need to use sample per epoch
    # see https://github.com/keras-team/keras/issues/5818

    # save tensorboard log if logdir is not None
    model.fit_generator(
        train_generator,
        epochs=1,
        steps_per_epoch=train_len // train_batch+1,
        validation_data=validation_generator,
        validation_steps=validation_len // test_batch+1,
        callbacks=[tensorboard, early_stopping if log_path is not None else early_stopping],
    )

    #TODO: check batch size
    val_score = model.evaluate_generator(validation_generator, validation_len // test_batch+1)
    print('val_score: ', val_score)

    test_score = model.evaluate_generator(test_generator, test_len// test_batch +1)
    print('test score 1: ', test_score)
    test_score = model.evaluate_generator(test_generator, test_len+1)
    print("test score 2", test_score)

    # save the model if dir is passed
    if save_model_path is not None:
        save_model(model, save_model_path)
        export_pb(model, save_model_path)

    return val_score, test_score

def save_model(model, path):
    try:
        model.save(path)
    except:
        print('cannot save model')
        pass
    return

def export_pb(model, path):
    print('exporting model to pb file')
    # print(model.output.op.name)
    saver = tf.train.Saver()
    saver.save(K.get_session(), path)
    return

def restore_model(model_path):
    model = load_model(model_path)
    num_layers = len(model.layers)
    print('Restored model from path ', model_path)
    print (model.summary())
    return model, num_layers


def main(_):
    # train(GENERAL_SETTING['image_dir'], 20, 10, 8, 'inception_v3')
    # image_lists = get_image_lists(GENERAL_SETTING['image_dir'], 20, 10)
    # print(image_lists)
    # train_generator, validation_generator, test_generator = get_generators()
    test_len = int(842 * 0.2)
    model = restore_model('/home/long/keras_resnet_2.h5')

    # score = model.evaluate_generator(test_generator, test_len+1)
    # print("score", score)
    export_pb(model, '')

if __name__ == '__main__':
      tf.app.run(main=main)
