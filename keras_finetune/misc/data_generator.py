from keras_impl.utils import get_random_decoded_images


class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self):
    return

  def generate(self, sess, image_lists, batch_size, category,
                          image_dir, jpeg_data_tensor, decoded_image_tensor):
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset

          (data, ground_truth, _) = get_random_decoded_images(sess,
                                                              image_lists, batch_size, category,
                                                              image_dir, jpeg_data_tensor,
                                                              decoded_image_tensor)
          yield data, ground_truth
