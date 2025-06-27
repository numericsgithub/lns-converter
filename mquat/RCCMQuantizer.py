# -*- coding: utf-8 -*-

import tensorflow as tf

from .FlexPointQuantizerNoClip import FlexPointQuantizerNoClip
from .FlexPointQuantizer import FlexPointQuantizer
from .Quantizer import Quantizer
from .Variable import Variable
from .Utilities import to_variable, to_value
from .QuantizerBase import DEFAULT_DATATYPE


# quantization to specified points
# uses the Basis-Class for initialisation of parameters if needed
class RCCMQuantizer(Quantizer):
    """RCCM quantisizer

    the RCCM-quantisizer quantisizes the input to the nearest points specified
    in an array.

    Parameters:
        name (string):
            the name of the layer in the TensorFlow graph.
        q_array (numpy float32 array):
            the quantisation points.
        leak_clip (float):
            leak factor for backpropagation.
            when values are outside the quantsiation range (clipped to range)
            the gradient is multiplied by leak_clip.
        dtype (tf.dtypes.DType):
            datatype of the layers's operations.
            default is float32.

    Attributes:
        q_array (tf.Variable float32):
            the quantisation points.
        leak_clip (float):
            leak factor for backpropagation.
            when values are outside the quantsiation range (clipped to range)
            the gradient is multiplied by leak_clip.
    """
    def __init__(self, name, q_array: tf.Variable, adder_type, find_coeff_set_py_func, leak_clip=0.0, pre_flex_bits=9, pre_keep_factor=0, dtype=DEFAULT_DATATYPE):
        super().__init__(name, dtype)
        self.__adder_type = adder_type
        self.leak_clip = tf.cast(leak_clip, DEFAULT_DATATYPE)
        self.__find_coeff_set_py_func = find_coeff_set_py_func
        if pre_flex_bits != None:
            self.preflex = FlexPointQuantizerNoClip(name + "_pre_flex_point", total_bits=pre_flex_bits)
        else:
            self.preflex = lambda x: x
        self.sample_counter = tf.Variable(name=name + "_counter", initial_value=0, trainable=False, dtype=tf.int32)
        self.coeff = q_array


    # def getVariables(self):
    #     """get all variables of the layer.
    #
    #     Returns:
    #         (list of Varaiables):
    #             list contains the weight and the bias Variable.
    #     """
    #     variables = super().getVariables()
    #     variables.extend([self.coeff_var])
    #     return variables

    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = super().getQuantVariables()
        if isinstance(self.preflex, FlexPointQuantizer):
            variables.extend(self.preflex.getQuantVariables())
        variables.extend([self.coeff, self.sample_counter])
        return variables


    def reset(self):
        """
        Resets the Quantizer. The next inference triggers the coefficient search again.
        Returns:

        """
        self.sample_counter.assign(0)

    @tf.function
    def wrapit(self, y_tmp):
        for min, max, coeff in zip(self.coeff_mins, self.coeff_maxs, self.coeff_values):
            y_tmp = tf.where(tf.logical_and(y_tmp > min, y_tmp <= max), coeff, y_tmp)
        return y_tmp

    @tf.function(jit_compile=True)
    def quant_to_coeffs(self, inputs, q_array):
        q_reshaped = tf.reshape(q_array, [tf.shape(q_array)[0], 1])
        # reshape x to flat tensor
        x_tmp = tf.reshape(inputs, [-1])
        # build a grid of abs differences between x and q
        # find the the index of the minimal distance
        abs_diff = tf.abs(x_tmp - q_reshaped)
        min_index = tf.argmin(abs_diff, axis=0)
        # get the corresponding quantisation value
        y_tmp = tf.gather(q_array, min_index)
        # reshape the result back to its oroginal form
        y = tf.reshape(y_tmp, tf.shape(inputs))
        return y

    def quant(self, inputs):
        """quantization function

        applies the quantization to the input.

        Parameters:
            inputs (list of tensors):
                list of all input tensors.

        Returns:
            (tensor):
                output of the layer.
        """

        @tf.custom_gradient
        # custom function, that modifies the gradient computation in the
        # backward pass
        def _quant(inputs, q_array, min_value, max_value, leak_clip):
            self.sample_counter.assign_add(1)
            tf.cond(tf.equal(self.sample_counter, tf.constant(1)), lambda: tf.py_function(self.__find_coeff_set_py_func, inp=[self.__adder_type, self.preflex(inputs)], Tout=tf.float32), lambda: 0.0)
            #y_tmp = inputs
            #y_tmp = tf.cond(self.sample_counter > tf.constant(1), lambda: tf.py_function(self.wrapit, inp=[y_tmp], Tout=tf.float32), lambda: y_tmp)
            y = self.quant_to_coeffs(inputs, q_array)
            # define the gradient calculation
            def grad(dy): # , variables=None
                # test for every element of x if it is out of the bounds of
                # the quantisation range
                is_out_of_range = tf.logical_or(inputs < min_value, inputs > max_value)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return tf.where(is_out_of_range, leak_clip * dy, dy), tf.zeros_like(q_array), tf.zeros_like(
                    min_value), tf.zeros_like(max_value), tf.zeros_like(leak_clip)

            return y, grad
        return _quant(inputs, self.coeff, self.min_value, self.max_value, self.leak_clip)


    def get_config(self):
        config =super().get_config()
        config["type"] = "mquat.RCCMQuantizer"
        return config