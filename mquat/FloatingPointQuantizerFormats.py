# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from .Quantizer import Quantizer


def powInt(x, y):
    x = tf.cast(x, tf.int64)
    y = tf.cast(y, tf.int64)
    z = tf.pow(x, y)
    return z


def pow(x, y):
    x = tf.cast(x, tf.int64)
    y = tf.cast(y, tf.int64)

    y_neg_mask = tf.where(y < tf.cast(0, tf.int64), tf.cast(-1, tf.int64), tf.cast(1, tf.int64))
    y = y * y_neg_mask
    # Extreme cases like inf are not affected by this!
    # So check again and put dummies in for those numbers
    y = tf.where(y < tf.cast(0, tf.int64), tf.cast(1, tf.int64), y)  # replace INF, -INF and NaN with a dummy value
    z = tf.cast(tf.pow(x, y), tf.float32)  # tf.cast(tf.pow(2, tf.cast(exponent, tf.int32)), tf.float32)
    z = tf.where(y_neg_mask == -1, 1.0 / z, 1.0 * z)
    return z


class FloatingPointQuantizerFormats():


    def __init__(self, fp_quant, format_string:str):
        self.fp_quant = fp_quant
        self.format_string:str = format_string
        format_map = {"E4M3": self.__E4M3, # https://arxiv.org/abs/2209.05433
                      "E5M2": self.__E5M2,  # https://arxiv.org/abs/2209.05433
                      }
        format_range_map = {"E4M3": self.__E4M3_Range, # https://arxiv.org/abs/2209.05433
                            "E5M2": self.__E5M2_Range,  # https://arxiv.org/abs/2209.05433
                      }
        self.__format_function = format_map[format_string]
        self.__format_range_function = format_range_map[format_string]

    def formatRange(self):
        return self.__format_range_function()

    def formatQuantizedValues(self, inputs_quanted, sign, exponent, mantissa_quanted, zero_inputs_mask, inputs):
        """
        Implements the chosen floating point format. This simply means that the inputs are already quantized
        to inputs_quanted but the classification of the special value representations (INF, -INF, NaN, Zeros) are missing.
        Those are set in this function.
        Args:
            inputs_quanted: The quantized inputs.
            sign: The signs of all quantized floating points.
            exponent: The exponent of all quantized floating points.
            mantissa_quanted:
            zero_inputs_mask:
            inputs:

        Returns:

        """
        return self.__format_function(inputs_quanted, sign, exponent, mantissa_quanted, zero_inputs_mask, inputs)

    def __E4M3_Range(self):
        biggest_exponent = self.fp_quant.exponent_bias + 1  # the biggest exponent possible
        # biggest_exponent_shift = pow(2, self.fp_quant.exponent_bits) - self.fp_quant.exponent_bias - 1.0  # the biggest exponent possible
        max_value = pow(2, biggest_exponent) * (1 + 1 - pow(2, -self.fp_quant.mantissa_bits + 1))  # biggest and smallest representable value
        return -max_value, max_value

    def __E4M3(self, inputs_quanted, sign, exponent, mantissa_quanted, zero_inputs_mask, inputs):
        smallest_exponent = -self.fp_quant.exponent_bias  # the smallest exponent possible
        biggest_exponent = self.fp_quant.exponent_bias + 1  # the biggest exponent possible
        # biggest_exponent_shift = pow(2, self.fp_quant.exponent_bits) - self.fp_quant.exponent_bias - 1.0  # the biggest exponent possible
        max_value = pow(2, biggest_exponent) * (1 + 1 - pow(2, -self.fp_quant.mantissa_bits + 1))  # biggest and smallest representable value

        # # # All cases for the zero:
        # 1) The exponent is smaller than allowed -> The number is between the smallest representable negative and positive number
        inputs_quanted = tf.where(exponent < smallest_exponent, 0.0, inputs_quanted)  # Zero out everything smaller than the smallest number
        # 2) The exponent == smallest_exponent and mantissa is zero. This setting represents 0 and -0 depending on the sign bit
        inputs_quanted = tf.where(tf.logical_and(mantissa_quanted == 1.0, exponent == smallest_exponent), 0.0, inputs_quanted)  # The zero case where "0 000 0000 -> 0" and "1 000 0000 -> -0"
        # 3) The number was zero before quantization
        inputs_quanted = tf.where(zero_inputs_mask, 0.0, inputs_quanted)  # zero is zero and not some sort of inf, -inf or NaN which is calculated when calculating log(0)

        # BEWARE! If a value cannot be represented in the format because it is out of range:
        # The value has to be clipped
        inputs_quanted = tf.where(tf.logical_and(inputs > max_value, inputs != np.inf), max_value, inputs_quanted)
        inputs_quanted = tf.where(tf.logical_and(inputs < -max_value, inputs != -np.inf), -max_value, inputs_quanted)
        return inputs_quanted

    def __E5M2_Range(self):
        biggest_exponent = self.fp_quant.exponent_bias  # the biggest exponent possible
        # biggest_exponent_shift = pow(2, self.fp_quant.exponent_bits) - self.fp_quant.exponent_bias - 1.0  # the biggest exponent possible
        max_value = pow(2, biggest_exponent) * (1 + 1 - pow(2, -self.fp_quant.mantissa_bits))  # biggest and smallest representable value
        return -max_value, max_value

    def __E5M2(self, inputs_quanted, sign, exponent, mantissa_quanted, zero_inputs_mask, inputs):
        smallest_exponent = -self.fp_quant.exponent_bias  # the smallest exponent possible
        biggest_exponent = self.fp_quant.exponent_bias  # the biggest exponent possible
        # biggest_exponent_shift = pow(2, self.fp_quant.exponent_bits) - self.fp_quant.exponent_bias - 1.0  # the biggest exponent possible
        max_value = pow(2, biggest_exponent) * (1 + 1 - pow(2, -self.fp_quant.mantissa_bits))  # biggest and smallest representable value

        # # # All cases for the zero:
        # 1) The exponent is smaller than allowed -> The number is between the smallest representable negative and positive number
        inputs_quanted = tf.where(exponent < smallest_exponent, 0.0,
                                  inputs_quanted)  # Zero out everything smaller than the smallest number
        # 2) The exponent == smallest_exponent and mantissa is zero. This setting represents 0 and -0 depending on the sign bit
        inputs_quanted = tf.where(tf.logical_and(mantissa_quanted == 1.0, exponent == smallest_exponent),
                                  0.0,
                                  inputs_quanted)  # The zero case where "0 000 0000 -> 0" and "1 000 0000 -> -0"
        # 3) The number was zero before quantization
        inputs_quanted = tf.where(zero_inputs_mask, 0.0,
                                  inputs_quanted)  # zero is zero and not some sort of inf, -inf or NaN which is calculated when calculating log(0)

        # BEWARE! If a value cannot be represented in the format because it is out of range:
        # The value has to be clipped
        inputs_quanted = tf.where(tf.logical_and(inputs > max_value, inputs != np.inf), max_value, inputs_quanted)
        inputs_quanted = tf.where(tf.logical_and(inputs < -max_value, inputs != -np.inf), -max_value, inputs_quanted)

        # # # # All cases for inf
        # inputs_quanted = tf.where(tf.logical_and(exponent == biggest_exponent_shift, mantissa_quanted == 1.0),
        #                           tf.where(sign == 1.0, np.inf, -np.inf), inputs_quanted)
        #
        # # # # All cases for MIN and MAX
        # inputs_quanted = tf.where(tf.logical_and(inputs > max_value, inputs != np.inf), max_value, inputs_quanted)
        # inputs_quanted = tf.where(tf.logical_and(inputs < -max_value, inputs != -np.inf), -max_value, inputs_quanted)
        #
        # # # # All cases for NaN
        # inputs_quanted = tf.where(tf.logical_and(exponent == biggest_exponent_shift, mantissa_quanted != 1.0), np.nan, inputs_quanted)
        return inputs_quanted