import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
import mquat as mq
from keras.preprocessing.image import ImageDataGenerator

IMAGE_CROP_SIZE = 32
datagen = ImageDataGenerator(
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # #rotation_range=1.0,
        # # zoom_range=0.3

        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
    )

def py_augment(image):
    image = image.numpy()
    image = datagen.apply_transform(image, datagen.get_random_transform([32,32,3]))
    #image = tf.keras.preprocessing.image.random_shear(image, intensity=2, row_axis=0, col_axis=1, channel_axis=2)
    #image = tf.keras.preprocessing.image.random_zoom(image, zoom_range=(0.90, 1.0), row_axis=0, col_axis=1, channel_axis=2)
    # image = tf.keras.preprocessing.image.random_rotation(image, rg=15, row_axis=0, col_axis=1, channel_axis=2)
    # image = tf.keras.preprocessing.image.random_shift(image, wrg=0.1, hrg=0.1, row_axis=0, col_axis=1, channel_axis=2)
    return image

# IMAGE_CROP_SIZE = 32
#
# def py_augment(image):
#     image = image.numpy()
#     image = tf.keras.preprocessing.image.random_shear(image, intensity=5, row_axis=0, col_axis=1, channel_axis=2)
#     image = tf.keras.preprocessing.image.random_zoom(image, zoom_range=(0.90, 1.0), row_axis=0, col_axis=1, channel_axis=2)
#     return image

@tf.function
def augment(image, label):
    #image = tf.cast(image, dtype=tf.float32) / 255.0
    #image = tf.image.random_flip_left_right(image)
    image = tf.py_function(func=py_augment, inp=[image], Tout=[tf.float32])
    image = tf.squeeze(image)
    image = tf.ensure_shape(image, [32, 32, 3])
    #image = tf.py_function(func=py_augment, inp=[image], Tout=[tf.float32])
    #image = tf.squeeze(image)
    # r, g, b = tf.split(image, 3, axis=2)
    # r = (r - 0.5071) / 0.2673
    # g = (g - 0.4865) / 0.2564
    # b = (b - 0.4409) / 0.2762
    #image = tf.concat([r,g,b], axis=2)
    return image, tf.one_hot(label, 100, dtype=tf.float32)


@tf.function()
def augment_val(image, label):
    # image = tf.cast(image, dtype=tf.float32) / 255.0
    # r, g, b = tf.split(image, 3, axis=2)
    # r = (r - 0.5071) / 0.2673
    # g = (g - 0.4865) / 0.2564
    # b = (b - 0.4409) / 0.2762
    #image = tf.concat([r,g,b], axis=2)
    return image, tf.one_hot(label, 100, dtype=tf.float32)

@tf.function
def decode_tf_record_entry(entry):
    image = entry["image"]
    label = entry["label"]
    image = tf.cast(image, dtype=tf.float32) / 255.0
    r, g, b = tf.split(image, 3, axis=2)
    r = (r - 0.5071) / 0.2673
    g = (g - 0.4865) / 0.2564
    b = (b - 0.4409) / 0.2762
    image = tf.concat([r, g, b], axis=2)
    return image, label


def get_train_ds(TRAIN_BATCH_SIZE, train_record_path=None):
    train_dataset: tf.data.TFRecordDataset = tfds.load('cifar100', split='train', shuffle_files=True,
                                                       data_dir=train_record_path, download=True)
    train_dataset = train_dataset.map(decode_tf_record_entry).cache().shuffle(32)
    train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(TRAIN_BATCH_SIZE)
    train_dataset = train_dataset.prefetch(1)
    return train_dataset


def get_test_ds(TEST_BATCH_SIZE, test_record_path=None):
    test_dataset: tf.data.TFRecordDataset = tfds.load('cifar100', split='test', shuffle_files=True,
                                                      data_dir=test_record_path, download=True)
    test_dataset = test_dataset.map(decode_tf_record_entry)
    test_dataset = test_dataset.map(augment_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(TEST_BATCH_SIZE).cache().prefetch(1)
    return test_dataset

# import matplotlib.pyplot as plt
# test = get_train_ds(32).unbatch().as_numpy_iterator()
# for i in range(20):
#     test = get_train_ds(32).unbatch().as_numpy_iterator()
#     c = 0
#     for e in test:
#         c += 1
#         if c == 7:
#             print("asd", np.min(e[0]), np.max(e[0]))
#             plt.imshow(e[0])
#             plt.show()
#             break
# exit(0)