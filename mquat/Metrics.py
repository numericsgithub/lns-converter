# -*- coding: utf-8 -*-

import tensorflow as tf


def top_1_accuracy(t, y):
    """top 1 categorial accuracy
        
    calculates the top 1 accuracy (percentage of right predictions).
    
    Parameters:
        t (float32 tensor):
            target.
        y (float32 tensor):
            prediction.
            
    Returns:    
        (1d float32 tensor):
            accuracy.
    """
    # find the equal max values in t and y (correct prediction)
    correct_prediction = tf.equal(tf.argmax(t, axis=-1), tf.argmax(y, axis=-1))
    # calculate the accuracy by the mean along the batch dimension
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float64), axis=0)


def modified_top_1_accuracy(t, y):
    """top 1 categorial accuracy
        
    special metric that handles the case if more than one value of y is the
    maximum, if its the case the accuracy is divided by the number of maximal
    values.
    
    Parameters:
        t (float32 tensor):
            target.
        y (float32 tensor):
            prediction.
            
    Returns:
        (1d float32 tensor):
            accuracy.
    """
    # find the maximum along the last axis
    y_max = tf.reduce_max(y, axis=-1)
    y_max_exp = tf.expand_dims(y_max, axis=-1)
    # true for every element that matches the maximum
    is_equal_y_max = tf.equal(y, y_max_exp)
    # count the marked elements along the last axis (number of max predictions)
    num_y_max = tf.math.count_nonzero(is_equal_y_max, axis=-1, dtype=tf.float32)
    # convert t to bool tensor
    is_t_max = tf.equal(t, 1.0)
    # test if bool target and bool prediction have the same entries
    tmp = tf.cast(tf.logical_and(is_equal_y_max, is_t_max), tf.float32)
    # if one of the predictions is equal to the target is_correct is true
    is_correct = tf.reduce_max(tmp, axis=-1)
    # divide by the number of max predictions 
    # calculate the accuracy by the mean along the batch dimension
    return tf.reduce_mean(is_correct / num_y_max, axis=0)


def create_top_k_arruracy(k):
    """top k categorial accuracy
        
    calculates the top k accuracy.
    
    Parameters:
        t (float32 tensor):
            target.
        y (float32 tensor):
            prediction.
            
    Returns:
        (1d float32 tensor):
            accuracy.
    """
    def _top_k_accuracy(t, y):
        # find the index of the target
        index_t = tf.expand_dims(tf.argmax(t, axis=-1, output_type=tf.int32), 1)
        # find the maximum indices of the top k largest y entries (predictions)
        _, indices_y = tf.nn.top_k(y, k)
        # test if some of them is equal the target index
        tmp = tf.cast(tf.equal(indices_y, index_t), tf.float32)
        # true if one of the indices matches
        is_correct = tf.reduce_max(tmp, axis=-1)
        # calculate the accuracy by the mean along the batch dimension
        return tf.reduce_mean(is_correct, axis=0)
    _top_k_accuracy.__name__ = f"top_{k}_accuracy"
    return _top_k_accuracy