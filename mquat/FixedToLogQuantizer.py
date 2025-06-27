# -*- coding: utf-8 -*-

import tensorflow as tf
from .Quantizer import Quantizer


class FixedToLogQuantizer(Quantizer):
    """fixed point quantisizer
    
    it is using a signed fixed point representation and is s based on the 
    LinearQuantizer.
    the quantisation range and number of steps is calculated by the 
    two bitsizes.
    """
    def __init__(self, name, msb, lsb, expLsb):
        super().__init__(name=name)
        self.maybe_inversed_step_diff = tf.pow(2.0, tf.cast(-lsb, tf.float32))
        self.msb = msb
        self.lsb = lsb
        self.expLsb = expLsb


    def quant(self, inputs):
        """quantisation function

        applies the quantization to the input.

        Parameters:
            inputs (list of tensors):
                list of all input tensors.

        Returns:
            (tensor):
                the output of layer.
        """

        @tf.custom_gradient
        # custom function, that modifies the gradient computation in the
        # backward pass
        def _quant(inputs):
            inputs_zeros = tf.where(inputs == 0.0, 0.0, 1.0)
            lwx = tf.where(inputs != 0.0, tf.math.log(tf.abs(inputs)) / tf.math.log(2.0), 0.0)  # get log2(inputs)
            lwx = lwx * tf.pow(2.0, -self.lsb) # round log2 number
            lwx = tf.floor(lwx + 0.5)          # round log2 number
            lwxFloat = lwx * tf.pow(2.0, self.lsb)  # round log2 number

            wxFloat = tf.pow(2.0 , lwxFloat) # wxFloat = Math.Pow(2, lwxFloat); // rounding @lsb=-53, 2**x

            wxFloat = wxFloat * tf.pow(2.0, -self.expLsb) # wxFloat *= Math.Pow(2, -expLsb); // exact, alignment

            summand = wxFloat * inputs_zeros  # filter NaN values
            summand = tf.floor(summand + 0.5) # summand = (long)Math.Floor(wxFloat + 0.5f); // rounding @expLsb

            # define the gradient calculation
            def grad(dy):
                # test for every element of x if it is out of the bounds of
                # the quantisation range
                #is_out_of_range = tf.logical_or(inputs < self.min_value, inputs > self.max_value)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return dy#tf.where(is_out_of_range, self.leak_clip * dy, dy)

            return summand, grad

        return _quant(inputs)