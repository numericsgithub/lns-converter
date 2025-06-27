# -*- coding: utf-8 -*-

import tensorflow as tf
from .QNetBaseModel import QNetBaseModel
from .QuantizerBase import DEFAULT_DATATYPE


class QNetFunctionalModel(QNetBaseModel):
    """mquat functional model
    
    use this class to build a model in the form of a keras functional model.
    
    Parameters: 
        name (string):
            the name of the model in the TensorFlow graph.
        x (Keras Tensor):
            the keras input tensor of the model.
        y (Keras Tensor or dict of Keras Tensors):
            the output tensors of the model.
        target_shape (list of ints):
            target shape (y_true shape) of the model without the batch dimension
        dtype (tf.dtypes.DType):
            datatype of the models's operations and weights/variables.
            default is float32.  
    """
    def __init__(self, name, x, y, target_shape, dtype=DEFAULT_DATATYPE):
        y_shape = y[0].shape if isinstance(y, list) else y.shape
        super().__init__(name, x.shape[1:], y_shape, target_shape, inputs=x, outputs=y)