# -*- coding: utf-8 -*-

import tensorflow as tf
from .ComplexLayer import ComplexLayer
from .MetaAutofilter import MetaAutofilter
from .QuantizerBase import DEFAULT_DATATYPE


class MetaLayer(ComplexLayer):
    """meta layer

    the complex layer contains an activation function and optionally a
    batch normalization layer
    the functionality is created in the subclasses

    Parameters:
        name (string):
            the name of the layer in the TensorFlow graph.
        activation_func (function):
            actiavtion function of the layer.
            can be custom or from tf.nn.
        do_batch_norm (bool):
            true if batch nomalization is needed.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations and weights/variables.
            default is float32.

    Attributes:
        activation (ActivationLayer):
            activation layer
        batch_norm (BatchNormalization):
            applied before or after the activation function.
            can be None if not needed.
    """
    def __init__(self, name, activation_fn, do_batch_norm, dtype=DEFAULT_DATATYPE):
        super().__init__(name, activation_fn, do_batch_norm, dtype)
        self.meta_initializer = None
        self.inference_counter = tf.Variable(0, name=self.name+"_inference_counter", trainable=False, dtype=tf.int32)

    def reset(self):
        """
        Resets the filter. The next inference triggers the filter search again.
        Returns:

        """
        self.inference_counter.assign(0)

    def meta_build(self, input_shape):
        pass

    def meta_call(self, inputs, training):
        pass

    def build(self, input_shape):
        """build the variables of the layer

        called by TensorFlow before the first forward propagation.

        Parameters:
            input_shape (tuple of 4 ints):
                shape of the input tensor.
        """
        self.meta_build(input_shape)

    def call(self, inputs, training):
        """forward propagation

        apply the layer operation in the forward pass.

        Parameters:
            inputs (list):
                list of all input tensors.
            training (bool):
                True: if in training mode.

        Returns:
            (tensor):
                the output of the layer.
        """
        self.inference_counter.assign_add(1)
        # tf.cond(self.inference_counter == 1,
        #         lambda: self.meta_initializer.initial(inputs, training, self.meta_call),
        #         lambda: 0)
        #tf.print("call with", tf.reduce_sum(tf.where(inputs < 0.0, 1.0, 1.0)))
        meta_call_result = self.meta_call(inputs, training)
        return meta_call_result