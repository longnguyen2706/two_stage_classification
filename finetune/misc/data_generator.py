import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.platform import gfile

from finetune.misc.utils import get_decoded_image, add_jpeg_decoding
from utils import load_pickle


#TODO: fix the bug in shuffle=True
# see https://github.com/keras-team/keras/issues/9707
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, sess, list_images, labels, num_classes, image_dir,  jpeg_data_tensor, decoded_image_tensor, batch_size, shuffle=False):
        self.sess = sess
        self.batch_size = batch_size
        self.labels = labels
        self.list_images = list_images
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.image_dir = image_dir
        self.jpeg_data_tensor = jpeg_data_tensor
        self.decoded_image_tensor = decoded_image_tensor
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_images ) / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_x = self.list_images[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Generate data
        # print(indexes)
        batch_data, batch_labels = self.__data_generation(batch_x, batch_y)
        return batch_data, batch_labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.list_images = np.arange(len(self.list_images))
        if self.shuffle == True:
            np.random.shuffle(self.list_images)

    def __data_generation(self, batch_x, batch_y):
        # print (batch_x, batch_y)
        batch_images = []
        for x in batch_x:
            image_path = os.path.join(self.image_dir, x)
            image_data = gfile.FastGFile(image_path, 'rb').read()
            decoded_image_data = get_decoded_image(self.sess, image_data, self.jpeg_data_tensor, self.decoded_image_tensor)
            batch_images.append(decoded_image_data[0])

        return np.asarray(batch_images), keras.utils.to_categorical(batch_y, num_classes=self.num_classes)


def get_generators(model_info, split, image_dir, train_batch, test_batch):
    train_images = split['train_files']
    train_labels = split['train_labels']

    val_images = split['val_files']
    val_labels = split['val_labels']

    test_images = split['test_files']
    test_labels = split['test_labels']
    num_classes = len(split['class_names'])

    sess = tf.Session()
    with sess.as_default():
        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            model_info['input_width'], model_info['input_height'],
            model_info['input_depth'], model_info['input_mean'],
            model_info['input_std'])
        train_generator  = DataGenerator(sess, train_images, train_labels, num_classes, image_dir, jpeg_data_tensor, decoded_image_tensor, train_batch)
        val_generator = DataGenerator(sess, test_images, test_labels, num_classes, image_dir, jpeg_data_tensor, decoded_image_tensor, test_batch)
        test_generator = DataGenerator(sess, val_images, val_labels, num_classes, image_dir, jpeg_data_tensor, decoded_image_tensor, test_batch)
    return train_generator, val_generator, test_generator
