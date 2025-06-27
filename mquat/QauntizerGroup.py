# -*- coding: utf-8 -*-
import math

import tensorflow as tf
from tensorflow import Variable
import numpy as np

from .Utilities import to_variable, to_value
from .Quantizer import Quantizer

# TODO Add variable for the self.flex_inferences stuff
# TODO Deactivate the dynamic change of the bits
# TODO Add threshold (TODO count outlier)
# TODO Implement percentile keep stuff...

class PerChannelQuantizer(Quantizer):
    """fixed point quantisizer with self adjusting bits_before and bits_after

    it is using a signed fixed point representation and is s based on the
    Quantizer.
    the quantisation range and number of steps is calculated by the
    two bitsizes.

    Parameters:
        name (string):
            the name of the layer in the TensorFlow graph.
        total_bits (int):
            the amount of bits for the fixed point quantisizer
        leak_clip (float):
            leak factor for backpropagation.
            applied when values are clipped to quantization range.
        dtype (tf.dtypes.DType):
            datatype of the layers's operations.
            default is float32.
    """
    total_bits: tf.Variable
    flex_inferences: tf.Variable
    std_keep_factor: tf.Variable

    ÜBER_SHIFT = 0

    def __init__(self, name, create_quanziter_func, split_axis, splits, debug=False, dtype=tf.float32):
        super().__init__(name, dtype=dtype)
        self.all_per_channel_quantizers = []
        self.splits = splits#tf.Variable(initial_value=-1,trainable=False)
        self.split_axis = split_axis#tf.Variable(initial_value=-1,trainable=False)
        #self.is_init = False #tf.Variable(initial_value=False,trainable=False)
        self.create_quanziter_func = create_quanziter_func
        self.debug = debug
        for i in range(self.splits):
            self.all_per_channel_quantizers.append(self.create_quanziter_func(self.name, i))
        # self.reset()


    def reset(self):
        """
        Resets the Quantizer. The next inference triggers the coefficient search again.
        Returns:

        """
        # TODO Reset ALL THE STUFF
        raise Exception("RESET NOT IMPLEMENTED!")


    def call(self, inputs, quantisize=None):
        """forward propagation

        if the quantisizer is active it calls the quant function on the input
        and returns the result else it returns the input.

        Parameters:
            inputs (list of tensors):
                list of all input tensors.

        Returns:
            (tensor):
                quantisized imput.
        """
        if self.isQuantisizing():
            split_inputs = tf.split(inputs, self.splits, axis=self.split_axis)
            for cur_chan, cur_quan in enumerate(self.all_per_channel_quantizers):
                split_inputs[cur_chan] = cur_quan(split_inputs[cur_chan], quantisize)
                #tf.print(self.name, cur_chan, split_inputs[cur_chan])
            result = tf.concat(split_inputs, axis=self.split_axis)
            return result
        else:
            return inputs