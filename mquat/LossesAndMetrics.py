# -*- coding: utf-8 -*-

import tensorflow as tf


class CustomMSE(tf.keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        # Get prediction value at the gt index. Later just called like "the value".
        # "the value" here is of very much importantness my comrade!!
        pred_values_at_truth_indexes = tf.gather(y_pred, tf.argmax(y_true, axis=-1), axis=1, batch_dims=1)

        # Aight! Get amount of predictions higher than the value -> predicted classes higher rated then the gt
        # This tells us something like: The pred for gt is at the 7th highest place
        pred_values_at_truth_indexes = tf.broadcast_to(tf.expand_dims(pred_values_at_truth_indexes, -1),
                                                       tf.shape(y_pred))
        greater_preds = tf.where(y_pred > pred_values_at_truth_indexes, 1.0, 0.0)
        equal_preds = tf.where(y_pred == pred_values_at_truth_indexes, 1.0, 0.0)

        pred_is_incorrect = tf.reduce_max(greater_preds, axis=-1)
        pred_is_not_top1 = tf.broadcast_to(tf.expand_dims(pred_is_incorrect, -1), tf.shape(y_pred))
        pred_is_top1 = tf.where(pred_is_not_top1 == 1.0, 0.0, 1.0)  # flip ones with zeros

        # correct the tile breaks: Everytime the pred is correct but has tie breaks -> the tie breaks are raised so they get punished more!
        pred_is_correct_with_tie_break = tf.where(pred_is_top1 == equal_preds, equal_preds, 0.0)
        pred_is_correct_with_tie_break = tf.where(pred_is_correct_with_tie_break == y_true, 0.0,
                                                  pred_is_correct_with_tie_break)
        pred_correction_factor1 = tf.where(pred_is_correct_with_tie_break == 1.0, 10000.5, 1.0)

        # Get prediction value at the gt index. Later just called like "the value".
        # "the value" here is of very much importantness my comrade!!
        pred_values_at_truth_indexes = tf.gather(y_pred, tf.argmax(y_true, axis=-1), axis=1, batch_dims=1)

        # Aight! Get amount of predictions higher than the value -> predicted classes higher rated then the gt
        # This tells us something like: The pred for gt is at the 7th highest place
        pred_values_at_truth_indexes = tf.broadcast_to(tf.expand_dims(pred_values_at_truth_indexes, -1),
                                                       tf.shape(y_pred))
        greater_preds = tf.where(y_pred > pred_values_at_truth_indexes, 1.0, 0.0)
        equal_preds = tf.where(y_pred == pred_values_at_truth_indexes, 1.0, 0.0)
        amount_equal_preds = tf.reduce_sum(equal_preds, axis=-1)
        amount_equal_preds = tf.broadcast_to(tf.expand_dims(amount_equal_preds, -1),
                                             tf.shape(y_pred))  # amount of equal predictions: is always at least 1

        pred_is_incorrect = tf.reduce_max(greater_preds, axis=-1)
        pred_is_not_top1 = tf.broadcast_to(tf.expand_dims(pred_is_incorrect, -1), tf.shape(y_pred))
        pred_is_top1 = tf.where(pred_is_not_top1 == 1.0, 0.0, 1.0)  # flip ones with zeros

        pred_is_correct_with_tie_break = tf.where(pred_is_top1 == equal_preds, equal_preds, 0.0)
        # pred_is_correct_with_tie_break = tf.where(pred_is_correct_with_tie_break == y_true, 0.0, pred_is_correct_with_tie_break)
        pred_is_correct_with_tie_break = tf.where(pred_is_correct_with_tie_break == y_true,
                                                  pred_is_correct_with_tie_break, 0.0)
        pred_is_correct_with_tie_break = tf.where(amount_equal_preds == pred_is_correct_with_tie_break, 0.0,
                                                  pred_is_correct_with_tie_break)

        pred_correction_factor = tf.where(pred_is_correct_with_tie_break == 1.0, 0.000001, 1.0) * pred_correction_factor1
        y_pred = y_pred * pred_correction_factor
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, False)

def create_categorical_crossentropy_loss_strict(name, from_logits=False, epsilon=0.00000001):
    """create categorical crossentropy loss function

    returns a keras loss operation wrapped in a function to give it a name.

    Parameters:
        name (string):
            name of the metric in the model.
        from_logits (bool):
            True if y_pred is a logits tensor.
        epsilon (float):
            small value that is added to y_pred to prevent nan-errors.
            (prevents the case if all y_pred values are zero)
    Returns:
        (function):
            created loss function for usage in models.
    """
    def _loss_fn(y_true, y_pred):
        # Get prediction value at the gt index. Later just called like "the value".
        # "the value" here is of very much importantness my comrade!!
        pred_values_at_truth_indexes = tf.gather(y_pred, tf.argmax(y_true, axis=-1), axis=1, batch_dims=1)

        # Aight! Get amount of predictions higher than the value -> predicted classes higher rated then the gt
        # This tells us something like: The pred for gt is at the 7th highest place
        pred_values_at_truth_indexes = tf.broadcast_to(tf.expand_dims(pred_values_at_truth_indexes, -1),
                                                       tf.shape(y_pred))
        greater_preds = tf.where(y_pred > pred_values_at_truth_indexes, 1.0, 0.0)
        equal_preds = tf.where(y_pred == pred_values_at_truth_indexes, 1.0, 0.0)

        pred_is_incorrect = tf.reduce_max(greater_preds, axis=-1)
        pred_is_not_top1 = tf.broadcast_to(tf.expand_dims(pred_is_incorrect, -1), tf.shape(y_pred))
        pred_is_top1 = tf.where(pred_is_not_top1 == 1.0, 0.0, 1.0)  # flip ones with zeros

        # correct the tile breaks: Everytime the pred is correct but has tie breaks -> the tie breaks are raised so they get punished more!
        pred_is_correct_with_tie_break = tf.where(pred_is_top1 == equal_preds, equal_preds, 0.0)
        pred_is_correct_with_tie_break = tf.where(pred_is_correct_with_tie_break == y_true, 0.0, pred_is_correct_with_tie_break)
        pred_correction_factor1 = tf.where(pred_is_correct_with_tie_break == 1.0, 100.5, 1.0)

        # Get prediction value at the gt index. Later just called like "the value".
        # "the value" here is of very much importantness my comrade!!
        pred_values_at_truth_indexes = tf.gather(y_pred, tf.argmax(y_true, axis=-1), axis=1, batch_dims=1)

        # Aight! Get amount of predictions higher than the value -> predicted classes higher rated then the gt
        # This tells us something like: The pred for gt is at the 7th highest place
        pred_values_at_truth_indexes = tf.broadcast_to(tf.expand_dims(pred_values_at_truth_indexes, -1),
                                                       tf.shape(y_pred))
        greater_preds = tf.where(y_pred > pred_values_at_truth_indexes, 1.0, 0.0)
        equal_preds = tf.where(y_pred == pred_values_at_truth_indexes, 1.0, 0.0)
        amount_equal_preds = tf.reduce_sum(equal_preds, axis=-1)
        amount_equal_preds = tf.broadcast_to(tf.expand_dims(amount_equal_preds, -1),
                                             tf.shape(y_pred))  # amount of equal predictions: is always at least 1

        pred_is_incorrect = tf.reduce_max(greater_preds, axis=-1)
        pred_is_not_top1 = tf.broadcast_to(tf.expand_dims(pred_is_incorrect, -1), tf.shape(y_pred))
        pred_is_top1 = tf.where(pred_is_not_top1 == 1.0, 0.0, 1.0)  # flip ones with zeros

        pred_is_correct_with_tie_break = tf.where(pred_is_top1 == equal_preds, equal_preds, 0.0)
        # pred_is_correct_with_tie_break = tf.where(pred_is_correct_with_tie_break == y_true, 0.0, pred_is_correct_with_tie_break)
        pred_is_correct_with_tie_break = tf.where(pred_is_correct_with_tie_break == y_true,
                                                  pred_is_correct_with_tie_break, 0.0)
        pred_is_correct_with_tie_break = tf.where(amount_equal_preds == pred_is_correct_with_tie_break, 0.0,
                                                  pred_is_correct_with_tie_break)

        pred_correction_factor = tf.where(pred_is_correct_with_tie_break == 1.0, 0.00001, 1.0) * pred_correction_factor1
        y_pred = tf.cast(y_pred, tf.float32) * pred_correction_factor
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits)
    _loss_fn.__name__ = name
    return _loss_fn


def create_categorical_crossentropy_loss(name, from_logits=False, epsilon=0.00000001):
    """create categorical crossentropy loss function

    returns a keras loss operation wrapped in a function to give it a name.

    Parameters:
        name (string):
            name of the metric in the model.
        from_logits (bool):
            True if y_pred is a logits tensor.
        epsilon (float):
            small value that is added to y_pred to prevent nan-errors.
            (prevents the case if all y_pred values are zero)
    Returns:
        (function):
            created loss function for usage in models.
    """
    def _loss_fn(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32) + epsilon
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits)
    _loss_fn.__name__ = name
    return _loss_fn

def create_sparse_categorical_crossentropy_loss(name, from_logits, epsilon=0.0):
    """create sparse categorical crossentropy loss

    returns a keras loss operation wrapped in a function to give it a name.

    Parameters:
        name (string):
            name of the metric in the model.
        from_logits (bool):
            True if y_pred is a logits tensor.
        epsilon (float):
            small value that is added to y_pred to prevent nan-errors.
            (prevents the case if all y_pred values are zero)
    Returns:
        (function):
            created loss function for usage in models.
    """
    def _loss_fn(label_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32) + epsilon
        return tf.keras.losses.sparse_categorical_crossentropy(label_true, y_pred, from_logits)
    _loss_fn.__name__ = name
    return _loss_fn


def create_sparse_top_k_acc_metric(name, k):
    """create a sparse top k categorical accuracy metric

    returns a keras metric operation wrapped in a function to give it a name.

    Parameters:
        name (string):
            name of the metric in the model.
        k (int):
            number of elements to consider for the accuracy.
    Returns:
        (function):
            created metric function for usage in models.
    """
    def _metric_fn(label_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(label_true, y_pred, k)
    _metric_fn.__name__ = name
    return _metric_fn

def create_top_k_acc_metric(name, k):
    """create a top k categorical accuracy metric

    returns a keras metric operation wrapped in a function to give it a name.

    Parameters:
        name (string):
            name of the metric in the model.
        k (int):
            number of elements to consider for the accuracy.
    Returns:
        (function):
            created metric function for usage in models.
    """
    def _metric_fn(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)
    _metric_fn.__name__ = name
    _metric_fn.name = name
    return _metric_fn


def create_sparse_valid_top_k_acc(name, k):
    """create a sparse valid top k categorical accuracy metric

    returns a special metric that handles the case if multible y_pred values are
    equal and also partly in top k.
    the function handles this case as undecidable (invalid) and returns zero accuracy.

    Parameters:
        name (string):
            name of the metric in the model.
        k (int):
            number of elements to consider for the accuracy.
    Returns:
        (function):
            created metric function for usage in models.
    """
    def _metric_fn(label_true, y_pred):
        label_true = tf.cast(label_true, tf.int32)
        y_pred_index = tf.gather(y_pred, label_true, batch_dims=1)
        larger_equal_index = y_pred >= y_pred_index
        larger_equal_index_count = tf.math.count_nonzero(larger_equal_index, axis=-1)
        return tf.cast(larger_equal_index_count <= k, tf.float32)
    _metric_fn.__name__ = name
    return _metric_fn

def create_valid_top_k_acc(name, k):
    """create a valid top k categorical accuracy metric

    returns a special metric that handles the case if multible y_pred values are
    equal and also partly in top k.
    the function handles this case as undecidable (invalid) and returns zero accuracy.

    Parameters:
        name (string):
            name of the metric in the model.
        k (int):
            number of elements to consider for the accuracy.
    Returns:
        (function):
            created metric function for usage in models.
    """
    # use the sparse metric
    _sub_metric = create_sparse_valid_top_k_acc(name, k)
    def _metric_fn(y_true, y_pred):
        label_true = tf.argmax(y_true, axis=-1)[..., tf.newaxis]
        return _sub_metric(label_true, y_pred)
    _metric_fn.__name__ = name
    return _metric_fn


def create_sparse_multi_top_k_acc(name, k):
    """create a multi top k categorical accuracy metric

    returns a special metric that handles the case if multible y_pred values are
    equal and also partly in top k.
    the function handles this case by setting the accuracy to the proability of that case.

    Parameters:
        name (string):
            name of the metric in the model.
        k (int):
            number of elements to consider for the accuracy.
    Returns:
        (function):
            created metric function for usage in models.
    """
    def _metric_fn(label_true, y_pred):
        label_true = tf.cast(label_true, tf.int32)
        y_pred_min = tf.sort(y_pred, axis=-1, direction="DESCENDING")[..., k-1:k]
        is_in_true = tf.gather(y_pred >= y_pred_min, label_true, batch_dims=1)[..., 0]
        y_pred_index = tf.gather(y_pred, label_true, batch_dims=1)
        larger_equal_min = y_pred >= y_pred_index
        larger_equal_min_count = tf.math.count_nonzero(larger_equal_min, axis=-1)
        return tf.where(is_in_true, k / (tf.maximum(larger_equal_min_count, k)), 0.0)
    _metric_fn.__name__ = name
    return _metric_fn

def create_multi_top_k_acc(name, k):
    """create a sparse multi top k categorical accuracy metric

    returns a special metric that handles the case if multible y_pred values are
    equal and also partly in top k.
    the function handles this case by setting the accuracy to the proability of that case.
    (used to be called as create_modified_top_k_acc)

    Parameters:
        name (string):
            name of the metric in the model.
        k (int):
            number of elements to consider for the accuracy.
    Returns:
        (function):
            created metric function for usage in models.
    """
    # use the sparse metric
    _sub_metric = create_sparse_multi_top_k_acc(name, k)
    def _metric_fn(y_true, y_pred):
        label_true = tf.argmax(y_true, axis=-1)[..., tf.newaxis]
        return _sub_metric(label_true, y_pred)
    _metric_fn.__name__ = name
    return _metric_fn


def categorical_crossentropy_with_epsilon(y_true,
                             y_pred,
                             from_logits=False,
                             label_smoothing=0,
                             axis=-1, epsilon=0.00000001):
    """Computes the categorical crossentropy loss.
    Args:
    y_true: Tensor of one-hot true targets.
    y_pred: Tensor of predicted targets.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
      example, if `0.1`, use `0.1 / num_classes` for non-target labels
      and `0.9 + 0.1 / num_classes` for target labels.
    axis: Defaults to -1. The dimension along which the entropy is
      computed.

    Returns:
    Categorical crossentropy loss value.
    """
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred + epsilon, from_logits, label_smoothing, axis)

def create_top_k_accuarcy_fixed(name, k, is_strict=False):
    _metric_fn = lambda y_true, y_pred: top_k_accuarcy_fixed(y_true, y_pred, k, is_strict)
    _metric_fn.__name__ = name
    _metric_fn.name = name
    return _metric_fn

def create_top_k_hits(name, k, is_strict=False):
    _metric_fn = lambda y_true, y_pred: top_k_hits(y_true, y_pred, k, is_strict)
    _metric_fn.__name__ = name
    _metric_fn.name = name
    return _metric_fn

class Top_k_accuarcy_fixed(tf.keras.metrics.Metric):

    def __init__(self, name, k, is_strict=False, **kwargs):
        super(Top_k_accuarcy_fixed, self).__init__(name=name, **kwargs)
        self.k = k
        self.is_strict = is_strict
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.counter = self.add_weight(name='counter', initializer='zeros')


    def update_state(self, y_true, y_pred, sample_weight=None):
        values = top_k_accuarcy_fixed(y_true, y_pred, self.k, self.is_strict)
        self.sum.assign_add(tf.reduce_sum(values))
        self.counter.assign_add(tf.cast(tf.size(values), dtype=tf.float32))

    def result(self):
        return self.sum / self.counter

def top_k_accuarcy_fixed(truths, preds, k, is_strict=False):
    """top k categorial accuracy

    special metric that handles the case if more than one value of y is the
    maximum, if its the case the accuracy is divided by the number of maximal
    values.

    Parameters:
        truths (float tensor):
            ground truth.
        preds (float tensor):
            prediction.
        k (int tensor):
            amount of top predictions to look at

    Returns:
        (1d float32 tensor):
            top k accuracy.
    """
    # Get prediction value at the gt index. Later just called like "the value".
    # "the value" here is of very much importantness my comrade!!
    pred_values_at_truth_indexes = tf.gather(preds, tf.argmax(truths, axis=-1), axis=1, batch_dims=1)

    # Aight! Get amount of predictions higher than the value -> predicted classes higher rated then the gt
    # This tells us something like: The pred for gt is at the 7th highest place
    pred_values_at_truth_indexes = tf.broadcast_to(tf.expand_dims(pred_values_at_truth_indexes, -1), tf.shape(preds))
    greater_preds = tf.where(preds > pred_values_at_truth_indexes, 1.0, 0.0)

    greater_preds = tf.reduce_sum(greater_preds, axis=1)

    # Is the amount of predictions with a higher value higher than k?
    # Like: Your pred at gt is 0.6 and you have 3 preds with 0.7 and 4 preds with 0.8
    #       If you want the top 5 acc: It already failed then and it should be 0
    #       BECAUSE: There are more higher rated preds then 5 -> Your 0.6 pred is too far away from being the top pred
    # So is_pred_in_top_k is 1 for preds where the amount of higher preds is smaller k
    # In other words: Checking if there are more then k higher preds. If so -> top_k=0. Otherwise -> top_k=?
    #                 Therefore we build a tensor that is 0 or 1. Later we multiply it with the ? value and we done.
    is_pred_in_top_k = tf.cast(greater_preds < k, dtype=tf.float32)

    # Now, my dude/dudet, look at the full range of relevant preds. We do it like before. Take every prediction
    # greater OR equal to the pred at gt and count them (We did exactly the same before but with tf.greater instead).
    # acc_k = tf.map_fn(lambda x: tf.cast(tf.greater_equal(x[0], x[1]), dtype=tf.float32),
    #                   (preds, pred_values_at_truth_indexes), dtype=tf.float32)  # tf.greater_equal for each row
    acc_k = tf.where(preds >= pred_values_at_truth_indexes, 1.0, 0.0)
    acc_k = tf.reduce_sum(acc_k, axis=1)

    # Almost there pal!
    # Imagine! You want the top k=3, your preds are [0.1, 0.2, 0.3, 0.3, 0.3, 0.9] and gt [0, 0, 0, 0, 1, 0]
    # Now you have 4 preds for the top k=3 to look at (The last 4 with 0.3 or 0.9).
    # This means your acc for this case is NOT 1! It should be 3/4 because you could not decide between those
    # four cases where you have like 1 pred to look at you would land at values higher than 1 because you calculate
    # 3/1 in those cases. Well that would not make any sense! That's why we take the minimum of the result with 1.
    # This suppresses values higher then 1 because min(x, 1) <= 1
    acc_k = k / acc_k

    # When being strict, only the clear cases are counted. All those maybes are rubbish dont ya think?
    # With k=1 and preds = [0.9, 0.9 ,0.1] gt = [1.0, 0.0, 0.0] the acc would be 0.5 but with strict it is 0.0
    # Because there are two candidates for the top 1 metric. You can be strict and punish this with a rating of 0.0
    # or lenient and rate it 0.5
    acc_k = tf.cond(tf.cast(is_strict, tf.bool), lambda: tf.where(acc_k >= 1.0, 1.0, 0.0), lambda: acc_k)
    acc_k = tf.minimum(acc_k, 1)

    # Ok now get rid of everything that has more then k predictions above it.
    acc_k = acc_k * is_pred_in_top_k
    return acc_k

def top_k_hits(truths, preds, k, is_strict=False):
    """top k categorial accuracy

    special metric that handles the case if more than one value of y is the
    maximum, if its the case the accuracy is divided by the number of maximal
    values.

    Parameters:
        truths (float tensor):
            ground truth.
        preds (float tensor):
            prediction.
        k (int tensor):
            amount of top predictions to look at

    Returns:
        (1d float32 tensor):
            top k accuracy.
    """
    # Get prediction value at the gt index. Later just called like "the value".
    # "the value" here is of very much importantness my comrade!!
    pred_values_at_truth_indexes = tf.gather(preds, tf.argmax(truths, axis=-1), axis=1, batch_dims=1)

    # Aight! Get amount of predictions higher than the value -> predicted classes higher rated then the gt
    # This tells us something like: The pred for gt is at the 7th highest place
    greater_preds = tf.map_fn(lambda x: tf.cast(tf.greater(x[0], x[1]), dtype=tf.float32),
                              (preds, pred_values_at_truth_indexes), dtype=tf.float32)  # tf.greater for each row
    greater_preds = tf.reduce_sum(greater_preds, axis=1)

    # Is the amount of predictions with a higher value higher than k?
    # Like: Your pred at gt is 0.6 and you have 3 preds with 0.7 and 4 preds with 0.8
    #       If you want the top 5 acc: It already failed then and it should be 0
    #       BECAUSE: There are more higher rated preds then 5 -> Your 0.6 pred is too far away from being the top pred
    # So is_pred_in_top_k is 1 for preds where the amount of higher preds is smaller k
    # In other words: Checking if there are more then k higher preds. If so -> top_k=0. Otherwise -> top_k=?
    #                 Therefore we build a tensor that is 0 or 1. Later we multiply it with the ? value and we done.
    is_pred_in_top_k = tf.cast(greater_preds < k, dtype=tf.float32)

    # Now, my dude/dudet, look at the full range of relevant preds. We do it like before. Take every prediction
    # greater OR equal to the pred at gt and count them (We did exactly the same before but with tf.greater instead).
    acc_k = tf.map_fn(lambda x: tf.cast(tf.greater_equal(x[0], x[1]), dtype=tf.float32),
                      (preds, pred_values_at_truth_indexes), dtype=tf.float32)  # tf.greater_equal for each row
    acc_k = tf.reduce_sum(acc_k, axis=1)

    # Almost there pal!
    # Imagine! You want the top k=3, your preds are [0.1, 0.2, 0.3, 0.3, 0.3, 0.9] and gt [0, 0, 0, 0, 1, 0]
    # Now you have 4 preds for the top k=3 to look at (The last 4 with 0.3 or 0.9).
    # This means your acc for this case is NOT 1! It should be 3/4 because you could not decide between those
    # four cases where you have like 1 pred to look at you would land at values higher than 1 because you calculate
    # 3/1 in those cases. Well that would not make any sense! That's why we take the minimum of the result with 1.
    # This suppresses values higher then 1 because min(x, 1) <= 1
    acc_k = k / acc_k

    # When beeing strict, only the clear cases are counted. All those maybes are rubbish dont ya think?
    # With k=1 and preds = [0.9, 0.9 ,0.1] gt = [1.0, 0.0, 0.0] the acc would be 0.5 but with strict it is 0.0
    # Because there are two candidates for the top 1 metric. You can be strict and punish this with a rating of 0.0
    # or lenient and rate it 0.5
    acc_k_strict = tf.where(acc_k == 1.0, 1.0, 0.0)
    acc_k = tf.where(acc_k >= 0.0, 1.0, 0.0)

    # Ok now get rid of everything that has more then k predictions above it.
    acc_k = acc_k * is_pred_in_top_k
    acc_k_strict = acc_k_strict * is_pred_in_top_k

    # Then take the mean of all the predictions
    acc_k = tf.reduce_sum(acc_k)
    acc_k_strict = tf.reduce_sum(acc_k_strict)
    return acc_k_strict if is_strict else acc_k


# y_pred = [[0.0, 0.8, 0.8, 0.5], [0.1, 0.4, 0.8, 0.5], [0.8, 0.8, 0.8, 0.5], [0.8, 0.8, 0.7, 0.5], [0.7, 0.8, 0.7, 0.5]]
# y_true = [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
#
# # Get prediction value at the gt index. Later just called like "the value".
# # "the value" here is of very much importantness my comrade!!
# pred_values_at_truth_indexes = tf.gather(y_pred, tf.argmax(y_true, axis=-1), axis=1, batch_dims=1)
#
# # Aight! Get amount of predictions higher than the value -> predicted classes higher rated then the gt
# # This tells us something like: The pred for gt is at the 7th highest place
# pred_values_at_truth_indexes = tf.broadcast_to(tf.expand_dims(pred_values_at_truth_indexes, -1),
#                                                tf.shape(y_pred))
# greater_preds = tf.where(y_pred > pred_values_at_truth_indexes, 1.0, 0.0)
# equal_preds = tf.where(y_pred == pred_values_at_truth_indexes, 1.0, 0.0)
# amount_equal_preds = tf.reduce_sum(equal_preds, axis=-1)
# amount_equal_preds = tf.broadcast_to(tf.expand_dims(amount_equal_preds, -1), tf.shape(y_pred)) # amount of equal predictions: is always at least 1
#
# pred_is_incorrect = tf.reduce_max(greater_preds, axis=-1)
# pred_is_not_top1 = tf.broadcast_to(tf.expand_dims(pred_is_incorrect, -1), tf.shape(y_pred))
# pred_is_top1 = tf.where(pred_is_not_top1 == 1.0, 0.0, 1.0) # flip ones with zeros
#
# pred_is_correct_with_tie_break = tf.where(pred_is_top1 == equal_preds, equal_preds, 0.0)
# # pred_is_correct_with_tie_break = tf.where(pred_is_correct_with_tie_break == y_true, 0.0, pred_is_correct_with_tie_break)
# pred_is_correct_with_tie_break = tf.where(pred_is_correct_with_tie_break == y_true, pred_is_correct_with_tie_break, 0.0)
# pred_is_correct_with_tie_break = tf.where(amount_equal_preds == pred_is_correct_with_tie_break, 0.0, pred_is_correct_with_tie_break)
# print("pred_is_top1", pred_is_top1)
# print("equal_preds", equal_preds)
# print("amount_equal_preds", amount_equal_preds)
# print("pred_is_correct_with_tie_break", pred_is_correct_with_tie_break)
# print("pred_values_at_truth_indexes", pred_values_at_truth_indexes)
#
# pred_correction_factor = tf.where(pred_is_correct_with_tie_break == 1.0, 1.0, 1.0)
# print("loss corr", tf.keras.losses.categorical_crossentropy(y_true, tf.cast(y_pred, tf.float32) * pred_correction_factor, False))
#
# pred_correction_factor = tf.where(pred_is_correct_with_tie_break == 1.0, 0.9, 1.0)
# print("loss corr", tf.keras.losses.categorical_crossentropy(y_true, tf.cast(y_pred, tf.float32) * pred_correction_factor, False))
#
# pred_correction_factor = tf.where(pred_is_correct_with_tie_break == 1.0, 0.5, 1.0)
# print("loss corr", tf.keras.losses.categorical_crossentropy(y_true, tf.cast(y_pred, tf.float32) * pred_correction_factor, False))
#
# pred_correction_factor = tf.where(pred_is_correct_with_tie_break == 1.0, 0.1, 1.0)
# print("loss corr", tf.keras.losses.categorical_crossentropy(y_true, tf.cast(y_pred, tf.float32) * pred_correction_factor, False))
#
# pred_correction_factor = tf.where(pred_is_correct_with_tie_break == 1.0, 0.01, 1.0)
# print("loss corr", tf.keras.losses.categorical_crossentropy(y_true, tf.cast(y_pred, tf.float32) * pred_correction_factor, False))
#
# # pred_correction_factor = tf.where(pred_is_correct_with_tie_break == 1.0, 1.5, 1.0)
# # print("loss corr", tf.keras.losses.categorical_crossentropy(y_true, tf.cast(y_pred, tf.float32) * pred_correction_factor, False))
# #
# # pred_correction_factor = tf.where(pred_is_correct_with_tie_break == 1.0, 2.5, 1.0)
# # print("loss corr", tf.keras.losses.categorical_crossentropy(y_true, tf.cast(y_pred, tf.float32) * pred_correction_factor, False))
# #
# # pred_correction_factor = tf.where(pred_is_correct_with_tie_break == 1.0, 10.5, 1.0)
# # print("loss corr", tf.keras.losses.categorical_crossentropy(y_true, tf.cast(y_pred, tf.float32) * pred_correction_factor, False))
# #
# # pred_correction_factor = tf.where(pred_is_correct_with_tie_break == 1.0, 100.5, 1.0)
# # print("loss corr", tf.keras.losses.categorical_crossentropy(y_true, tf.cast(y_pred, tf.float32) * pred_correction_factor, False))