# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py

from mquat.QuantizerBase import DEFAULT_DATATYPE


def create_h5_dataset(data_path, dataset_name, tf_data_type, shape, buffer_size=1000):
    """create 5h dataset
    
    the function loads a h5 file picks a dataset by name from it and returns a 
    TensorFlow dataset.
    
    Parameters: 
        data_path (string):
            path to the h5 dataset.
        dataset_name (string):
            the name of the dataset in h5 file.
        tf_data_type (tf datatype):
            the type that the data shoud be cast to.
        shape (list of int):
            the shape that each data item in the dataset has (without batch size).   
        buffer_size(int):
            the number of data items that are preloaded (for better performance).
            
    Returns:
        tf.data.Dataset:
            TensorFlow dataset.
    """
    def _generator():
        # generator function that loads the dataset while buffering paths of it
        # load the h5 file
        h5_file = h5py.File(data_path, 'r')
        # pick the dataset by name
        dataset = h5_file[dataset_name]
        data_lenght = len(dataset)
        # start index of the buffer
        start_index = 0
        while True:
            # get the end index (ensure that is never bigger than the daset lenght)
            end_index = min(start_index + buffer_size, data_lenght)
            # indexing in the dataset loads the data from disk
            # put it into the numpy buffer array
            buffer = dataset[start_index:end_index]
            # yield all entries of the buffer
            for data in buffer:
                yield data
            # if the end of dataset is reached break the loop
            if end_index == data_lenght:
                break
            start_index = end_index
        h5_file.close()
    return tf.data.Dataset.from_generator(_generator, tf_data_type, tf.TensorShape(shape))


def normalize_dataset(dataset):
    """normalize dataset
    
    the function converts a dataset to float32 and divides its entries by 255.
    
    Parameters: 
        dataset (tf.dataset):
            the dataset to normalize.
            
    Returns:
        tf.data.Dataset:
            TensorFlow dataset.
    """
    def normalize_img(image, label):
        return tf.cast(image, DEFAULT_DATATYPE) / 255.0, label
    return dataset.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def one_hot_encode_dataset(dataset, num_classes):
    """one hot encode dataset
    
    the function converts integer lables to a one hot encoded float32 tensors.
    
    Parameters: 
        dataset (tf.dataset):
            the dataset to encode.
        num_classes (int):
            the number of classes.
            
    Returns:
        tf.data.Dataset:
            TensorFlow dataset.
    """
    def _one_hot(image, label):
        return image, tf.one_hot(label, num_classes, dtype=DEFAULT_DATATYPE)
    return dataset.map(_one_hot, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def extra_norm_dataset(dataset):
    def augment(image, label):
        # image = image - 0.5
        return image, label
    return dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def augment_dataset(dataset):
    def augment(image, label):
        # image = tf.image.random_jpeg_quality (image, 95, 100)
        # tf.print(tf.shape(image2))

        # image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_saturation(image, 0.7, 1.4)
        # image = tf.image.random_brightness(image, 0.04)
        image = tf.reshape(image, [32, 32, 3])
        return image, label
    return dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def prepare_dataset(dataset, num_classes, batch_size, shuffle_buffer_size=None, cache=False, normalize=True):
    """prepare dataset
    
    the function applies the folowing operations to the dataset:
        1. normalization
        2. one hot encoding
        3. shuffling
        4. batching
        
    Parameters: 
        dataset (tf.dataset):
            the dataset to prepare.
        num_classes (int):
            the number of classes.
        batch_size (int):
            the batch size for training or testing.
        shuffle_buffer_size (int):
            the size for the shuffle buffer, if None no shuffling is applied.
            
    Returns:
        tf.data.Dataset:
            TensorFlow dataset.
    """
    if normalize:
        dataset = normalize_dataset(dataset)
    dataset = one_hot_encode_dataset(dataset, num_classes)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size != None:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def prepare_dataset_extra(dataset, num_classes, batch_size, augment_cifar10=False, do_not_shuffle=False):
    """prepare dataset

    the function applies the folowing operations to the dataset:
        1. normalization
        2. one hot encoding
        3. shuffling
        4. batching

    Parameters:
        dataset (tf.dataset):
            the dataset to prepare.
        num_classes (int):
            the number of classes.
        batch_size (int):
            the batch size for training or testing.
        shuffle_buffer_size (int):
            the size for the shuffle buffer, if None no shuffling is applied.

    Returns:
        tf.data.Dataset:
            TensorFlow dataset.
    """
    normalize = True
    shuffle_buffer_size = int(batch_size)
    if do_not_shuffle:
        shuffle_buffer_size = None
    if normalize:
        dataset = normalize_dataset(dataset)
    dataset = one_hot_encode_dataset(dataset, num_classes)
    dataset = extra_norm_dataset(dataset)
    if augment_cifar10:
       dataset = augment_dataset(dataset)
    if shuffle_buffer_size != None:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def create_rand_int(min_val, max_val):
    """create rand int
    
    generates a TensorFlow operation to generate random integers.
    
    Parameters:
        min_val (int):
            lower limit.
        max_val (int):
            upper limit.
            
    Returns:
        tf.random.uniform:
            TensorFlow operation for random uniform initialisation.
    """
    return tf.random.uniform([], minval=min_val, maxval=max_val, dtype=tf.int32)


def apply_random_flip_dataset(dataset, flip_y=True, flip_x=True):
    """apply randmom flip dataset
    
    applies a random flip horizonatal/vertical to the dataset.
    
    Parameters:
        dataset (tf.dataset):
            dataset.
        flip_y (bool):
            if True flip y-axis (horizontal).
        flip_x (bool):
            if True flip x-axis (vertical).
            
    Returns:
        tf.data.Dataset:
            TensorFlow dataset.
    """
    def _map(image, *args):
        if flip_y == True and create_rand_int(0, 2) == 1:
            image = tf.image.flip_up_down(image)
        if flip_x == True and create_rand_int(0, 2) == 1:
            image = tf.image.flip_left_right(image)
        return (image, *args)
    return dataset.map(_map)


def apply_gaussian_noise_dataset(dataset, stddev, mean=0.0):
    """apply gaussian noise dataset
    
    adds gaussian noise to the dataset.
    
    Parameters:
        dataset (tf.dataset):
            dataset.
        stddev (float):
            standard deviation of the normal distribution.
        mean (float):
            mean of the normal distribution.
            
    Returns:
        tf.data.Dataset:
            TensorFlow dataset.
    """
    def _map(image, *args):
        noise = tf.random.normal(shape=image.shape, mean=mean, stddev=stddev, dtype=DEFAULT_DATATYPE)
        image = image + noise
        return (image, *args)
    return dataset.map(_map)


def apply_random_crop_dataset(dataset, min_offset_h, max_offset_h, min_offset_w, max_offset_w):
    """apply random crop dataset
    
    expands the input images by the max offset horizonatal and vertical,
    then it selects a random part of the image (same size as the input)
    and returns it.
    
    Parameters:
        dataset (tf.dataset):
            dataset.
        min_offset_h (int):
            minimal offset vertical to pick the subimage.
        max_offset_h (int):
            minimal offset vertical to pick the subimage.
        min_offset_w (int):
            minimal offset horizontal to pick the subimage.
        max_offset_w (int):
            maximal offset horizontal to pick the subimage.
            
    Returns:
        tf.data.Dataset:
            TensorFlow dataset
    """
    def _map(image, *args):
        shape = image.shape
        image = tf.image.resize(image, (shape[0] + max_offset_h, shape[1] + max_offset_w))
        rand_offset_h = create_rand_int(min_offset_h, max_offset_h)
        rand_offset_w = create_rand_int(min_offset_w, max_offset_w)
        image = tf.image.crop_to_bounding_box(image, rand_offset_h, rand_offset_w, shape[0], shape[1])
        return (image, *args)
    return dataset.map(_map)


def apply_random_rotation_90_dataset(dataset, steps=[0, 1, 2, 3]):
    """apply random rotation 90 dataset
    
    rotates the images in 90 degrees steps.
    
    Parameters:
        dataset (tf.dataset):
            dataset.
        steps (list of int):
            possible steps to apply.
            
    Returns:
        tf.data.Dataset:
            TensorFlow dataset.
    """
    def _map(image, *args):
        rand_steps = tf.gather(steps, create_rand_int(0, len(steps)))
        image = tf.image.rot90(image, rand_steps)
        return (image, *args)
    return dataset.map(_map)


def plot_results(model, images, labels, images_x, images_y):
    """plot results

    plot the input images with labeled predictions in a matplotlib figure.

    Parameters:
        model (tf.keras.Model):
            neural net model.
        images (tf.tensor):
            example images.
        labels (tf.tensor):
            One Hot coded labels.
        images_x (int):
            number of images in x direction.
        images_y (int):
            number of images in y direction.
    """

    for i in range(images_x * images_y):
        plt.subplot(images_x, images_y, i + 1)
        plt.tight_layout()
        plt.imshow(images[i])
        y_i = model.predict(np.expand_dims(images[i], axis=0))
        plt.title(f"t:{np.argmax(labels[i])},y:{np.argmax(y_i)}")
        plt.xticks([])
        plt.yticks([])
        i += 1

def plot_results_dataset(model, dataset, images_x, images_y):
    """plot results
    
    plot the dataset images with labeled predictions in a matplotlib figure.
    
    Parameters:
        model (tf.keras.Model):
            neural net model.
        dataset (tf.dataset):
            dataset.
        images_x (int):
            number of images in x dierection.
        images_y (int):
            number of images in y dierection.
    """
    data = dataset.unbatch().take(images_x * images_y)
    i = 0
    for (x_i, t_i) in data:
        plt.subplot(images_x, images_y, i+1)
        plt.tight_layout()
        plt.imshow(x_i)
        y_i = model.predict(np.expand_dims(x_i, axis=0))
        plt.title(f"t:{t_i},y:{np.argmax(y_i)}")
        plt.xticks([])
        plt.yticks([])
        i += 1

def create_confusion_matrix(model, dataset, num_classes):
	# create a integer confusion matrix
    matrix = np.zeros([num_classes, num_classes], dtype=np.int64)
    # iterate over the batched dataset
    for image_batch, true_class_index_batch in dataset:
		# convert the dataset entry (TensorFlow tensor to numpy arrays)
		# prevents memory leaks
        image_batch = image_batch.numpy()
        true_class_index_batch = true_class_index_batch.numpy()
        # predict the batch
        y_pred_batch = model.predict(image_batch)
        # find the index of the maximum entries in the batch
        pred_class_index_batch = np.argmax(y_pred_batch, -1)
        # iterate the entries and add them to the confusion matrix by the index
        for true_class_index, pred_class_index in zip(true_class_index_batch, 
                                                      pred_class_index_batch):
            matrix[true_class_index, pred_class_index] += 1        
    return matrix