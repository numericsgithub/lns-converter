# -*- coding: utf-8 -*-

import tensorflow as tf
from .Layer import Layer
from .QuantizerBase import DEFAULT_DATATYPE


class SoftmaxLayer(Layer):
    """softmax layer

    the softmax layer applies softmax. That simple.
    Parameters:
        name (string):
            the name of the layer in the TensorFlow graph.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations.
            default is float32.
    """

    def __init__(self, name, dtype=DEFAULT_DATATYPE):
        super().__init__(name, dtype=dtype)

    def call(self, inputs):
        """forward propagation

        apply the layer operation in the forward pass.

        Parameters:
            inputs (list):
                list of all input tensors.

        Returns:
            (tensor):
                the output of the layer.
        """
        tmp = tf.keras.layers.Softmax()(inputs)
        return self.quant_out(tmp)

    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.SoftmaxLayer"
        return config