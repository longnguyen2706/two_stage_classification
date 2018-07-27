from __future__ import absolute_import

import os

import tensorflow as tf
from keras import backend as K
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.metrics import categorical_accuracy as accuracy
from keras.metrics import mae as mean_loss
from slim_utils import *

# slim = tf.contrib.slim
# from slim.nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
model_info = {
   'input_width':299,
    'input_height': 299,
    'input_depth':3,
    'input_mean': 128,
    'input_std':128
}

GENERAL_SETTING = {
    'bottleneck_dir': '/tmp/bottleneck',
    'logits_dir': 'tmp/logits',
    'early_stopping_n_steps': 5,
    'batch_size': 8,
    'eval_step_interval': 100,
    'final_tensor_name': 'final_result',
    'flip_left_right': False,
    'output_labels': '/tmp/output_labels.txt',
    'print_misclassified_test_images': True,
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
log_dir = '/home/long/logdir/keras/'

def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
      result_tensor: The new final node that produces results.
      ground_truth_tensor: The node we feed ground truth data
      into.

    Returns:
      Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_tensor, logits=result_tensor)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    return evaluation_step,cross_entropy_mean, prediction

def create_model_graph(model_info):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Args:
      model_info: Dictionary containing information about the model architecture.

    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
            bottleneck_tensor = tf.reshape(bottleneck_tensor, [1,model_info['bottleneck_tensor_size']])
    return graph, bottleneck_tensor, resized_input_tensor


def main(_):
    # Needed to make sure the logging output is visible.
    # See https://github.com/tensorflow/tensorflow/issues/3047
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare necessary directories that can be used during training
    # prepare_file_system()

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(GENERAL_SETTING['image_dir'], GENERAL_SETTING['testing_percentage'],
                                     GENERAL_SETTING['validation_percentage'])
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         FLAGS.image_dir +
                         ' - multiple classes are needed for classification.')
        return -1

    sess = tf.Session()
    K.set_session(sess)

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(64, input_shape=(2048,), activation='relu')(x)
    # x = Dropout(0.2)(x)

    predictions = Dense(10, activation='softmax')(x)

    #
    # model = Model(input=base_model.input, outputs=predictions)

    input = base_model.input
    labels = tf.placeholder(tf.float32, shape=(None, 10))

    from keras.objectives import categorical_crossentropy
    loss = tf.reduce_mean(categorical_crossentropy(labels, predictions))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    acc_ops = accuracy(labels, predictions)
    loss_ops = mean_loss(labels, predictions)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)



    # jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding()
    with sess.as_default():
        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            model_info['input_width'], model_info['input_height'],
            model_info['input_depth'], model_info['input_mean'],
            model_info['input_std'])

        # with slim.arg_scope(inception_resnet_v2_arg_scope()):
        #     logits, end_points = inception_resnet_v2(images, num_classes=dataset.num_classes, is_training=True)

        evaluation_step, cross_entropy_value, prediction = add_evaluation_step(
            predictions, input)

        for i in range(100):
            (train_data, train_ground_truth, _) =get_random_decoded_images(sess,
                image_lists, 16, 'training',
                GENERAL_SETTING['image_dir'], jpeg_data_tensor, decoded_image_tensor)

            # print (train_data.size, train_ground_truth.size)
            # train_data = np.array(train_data)
            # print(train_data.shape)
            train_step.run(feed_dict={input: train_data, labels: train_ground_truth})

            # acc_value = acc_ops.eval(feed_dict={input: train_data, labels: train_ground_truth})
            # loss_value = loss_ops.eval(feed_dict={input: train_data, labels: train_ground_truth})


            # print (acc_value, loss_value)


            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy_value])

            print(train_accuracy, cross_entropy_value)
            # tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
            #                 (datetime.now(), i, acc_value* 100))
            # tf.logging.info('%s: Step %d: Cross entropy = %f' %
            #                 (datetime.now(), i, loss_value))


            # print (i, "-", train_bottlenecks)



if __name__ == '__main__':
    tf.app.run(main=main)
