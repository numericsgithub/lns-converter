import tensorflow as tf
import tensorflow_datasets as tfds

from mquat.Datasets.ImageNetPre import preprocess_image
from mquat.QuantizerBase import DEFAULT_DATATYPE

IMAGE_CROP_SIZE = 224

TINY_IMAGENET_SIZE = 256
TINY_IMAGENET_CLASS_NUM = 10
TINY_IMAGE_CROP_SIZE = 224

def augment_p0(image, label):
    return tf.cast(image, tf.uint8), label # image, label # tf.cast(image, tf.float16), label


# def augment_p2(image, label):
#     return image, label

#@tf.function(jit_compile=True, input_signature=[tf.TensorSpec(shape=[IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3], dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int64)])
def augment_p2(image, label):
    image = data_generator_train.random_transform(image)

    # image = tf.ensure_shape(image, [IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3])
    return image, label# tf.cast(image, dtype=DEFAULT_DATATYPE), tf.one_hot(label, 1000, dtype=DEFAULT_DATATYPE)



def py_augment(image, label):
  func = tf.numpy_function(augment_p2, [image, label], [tf.float32, tf.int64])
  return func



@tf.function
def augment_val_filter(image, label):
    smallersize = tf.reduce_min(tf.shape(image)[0:2])
    return tf.greater_equal(smallersize, 224)


@tf.function
def decode_tf_record_entry(entry):
    image = entry["image"]
    label = entry["label"]
    return image, label+1


def get_train_ds(TRAIN_BATCH_SIZE, train_record_path=None, cache=False, image_net_classes=1001, fast_mode=True) -> tf.data.TFRecordDataset: # default is os.path.join(os.path.expanduser('~'), 'tensorflow_datasets')

    # @tf.function(jit_compile=False, input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8),
    #                                                  tf.TensorSpec(shape=(), dtype=tf.int64)])
    # def augment_p1(image, label):
    #     with tf.device("GPU"):
    #         image = tf.cast(image, DEFAULT_DATATYPE)
    #         # tf.io.write_file(r'C:\Users\fdai0217\Documents\GitHub\Cluster40\src\mquat\Datasets\test0.jpg', tf.image.encode_jpeg(tf.cast(image, tf.uint8), quality=100, format='rgb'))
    #
    #         smallersize = tf.reduce_min(tf.shape(image)[0:2])
    #         resizefactor = tf.random.uniform([], minval=256, maxval=400, dtype=tf.float32) / tf.cast(smallersize,
    #                                                                                                  dtype=tf.float32)
    #         h = tf.cast(tf.cast(tf.shape(image)[0], dtype=tf.float32) * resizefactor, dtype=tf.int32)
    #         w = tf.cast(tf.cast(tf.shape(image)[1], dtype=tf.float32) * resizefactor, dtype=tf.int32)
    #         image = tf.image.resize(image, [h, w])
    #
    #         # tf.io.write_file(r'C:\Users\fdai0217\Documents\GitHub\Cluster40\src\mquat\Datasets\test3.jpg', tf.image.encode_jpeg(tf.cast(image, tf.uint8), quality=100, format='rgb'))
    #         image = tf.image.random_crop(image, size=[IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3])
    #
    #         # tf.io.write_file(r'C:\Users\fdai0217\Documents\GitHub\Cluster40\src\mquat\Datasets\test4.jpg', tf.image.encode_jpeg(tf.cast(image, tf.uint8), quality=100, format='rgb'))
    #         return image, label  # tf.cast(image, tf.float32), tf.cast(label, tf.int32) # image, label # tf.cast(image, tf.float16), label
    #
    # @tf.function(jit_compile=True,
    #              input_signature=[tf.TensorSpec(shape=[IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3], dtype=tf.float32),
    #                               tf.TensorSpec(shape=(), dtype=tf.int64)])
    # def augment_p3(image, label):
    #     # image = data_generator_train.random_transform(image)
    #
    #     image = tf.image.random_flip_left_right(image)
    #     image = tf.image.random_saturation(image, 0.7, 1.4)
    #     image = tf.image.random_brightness(image, 1.0)
    #     r, g, b = tf.split(image, 3, axis=2)
    #     r = ((r / 255.0) - 0.485) / 0.229
    #     g = ((g / 255.0) - 0.456) / 0.224
    #     b = ((b / 255.0) - 0.406) / 0.225
    #     image = tf.concat([r, g, b], axis=2)
    #     return image, tf.one_hot(label, image_net_classes, dtype=DEFAULT_DATATYPE)
    #
    # @tf.function(jit_compile=True)
    # def augment_p4(image, label):
    #     r, g, b = tf.split(image, 3, axis=2)
    #     r = ((r / 255.0) - 0.485) / 0.229
    #     g = ((g / 255.0) - 0.456) / 0.224
    #     b = ((b / 255.0) - 0.406) / 0.225
    #     image = tf.concat([r, g, b], axis=2)
    #     return tf.cast(image, dtype=DEFAULT_DATATYPE), tf.one_hot(label, image_net_classes, dtype=DEFAULT_DATATYPE)
    print("IMAGE NET TRAIN DATASET AUGMENTATION: FAST_MODE =",fast_mode)

    @tf.function()
    def augment_train(image, label):
        image = preprocess_image(image, 224, 224, is_training=True, fast_mode=fast_mode)
        return tf.cast(image, dtype=DEFAULT_DATATYPE), tf.one_hot(label, image_net_classes, dtype=DEFAULT_DATATYPE)

    train_dataset: tf.data.TFRecordDataset = tfds.load('imagenet2012', split='train', shuffle_files=True,
                                                       data_dir=train_record_path, download=False)
    train_dataset = train_dataset.map(decode_tf_record_entry, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if cache:
        train_dataset = train_dataset.map(augment_p0, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.cache("/run/determined/workdir/shared/datasets/img2012_train7")
    train_dataset = train_dataset.map(augment_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.map(augment_p2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.map(py_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.map(augment_p3, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.map(lambda image,label: tf.py_function(func=augment_p3, inp=[image,label], Tout=[DEFAULT_DATATYPE, DEFAULT_DATATYPE]), num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).prefetch(1)
    #train_dataset = train_dataset.map(augment_p4, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


    # if cache:
    #     train_dataset = train_dataset.cache("/run/determined/workdir/shared/datasets/img2012_train5")
    train_dataset = train_dataset.shuffle(128).batch(TRAIN_BATCH_SIZE) #   .shuffle(128)
    return train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


def get_test_ds(TEST_BATCH_SIZE, test_record_path=None, cache=False, image_net_classes=1001) -> tf.data.TFRecordDataset:

    @tf.function()
    def augment_val(image, label):
        image = preprocess_image(image, 224, 224, is_training=False)
        return tf.cast(image, dtype=DEFAULT_DATATYPE), tf.one_hot(label, image_net_classes, dtype=DEFAULT_DATATYPE)

    test_dataset: tf.data.TFRecordDataset = tfds.load('imagenet2012', split='validation', shuffle_files=False,
                                                      data_dir=test_record_path, download=False)
    test_dataset = test_dataset.map(decode_tf_record_entry)
    # test_dataset = test_dataset.map(augment_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(augment_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(TEST_BATCH_SIZE)
    if cache:
        test_dataset = test_dataset.cache()
    else:
        test_dataset = test_dataset
    return test_dataset.prefetch(tf.data.AUTOTUNE)

