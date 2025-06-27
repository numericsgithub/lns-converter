# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from .FloatingPointQuantizerFormats import FloatingPointQuantizerFormats
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
    y = tf.where(y < tf.cast(0, tf.int64), tf.cast(1, tf.int64), y) # replace INF, -INF and NaN with a dummy value
    z = tf.cast(tf.pow(x, y), tf.float32)
    z = tf.where(y_neg_mask == -1, 1.0 / z, 1.0 * z)
    return z

class FloatingPointQuantizer(Quantizer):

    def __init__(self, name, format_type_string:str, channel_wise_scaling=False, scale_inputs=True):
        super(FloatingPointQuantizer, self).__init__(name, channel_wise_scaling=channel_wise_scaling, scale_inputs=scale_inputs)
        self.exponent_bits = tf.Variable(0.0, trainable=False, name=name+"_exponent_bits")
        self.mantissa_bits = tf.Variable(0.0, trainable=False, name=name+"_mantissa_bits")
        self.exponent_bias = tf.Variable(0.0, trainable=False, name=name+"_exponent_bias")
        self.fp_format = FloatingPointQuantizerFormats(self, format_type_string)


    def setParams(self, exponent_bits=None, mantissa_bits=None, scale=None):
        if exponent_bits != None:
            self.exponent_bits.assign(exponent_bits)
        if mantissa_bits != None:
            self.mantissa_bits.assign(mantissa_bits)
        self.exponent_bias.assign(pow(2, self.exponent_bits - 1.0) - 1.0)
        if scale != None:
            self.scale.assign(scale)
        min_value, max_value = self.fp_format.formatRange()
        self.min_value = min_value
        self.max_value = max_value

    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.Quantizer.FloatingPointQuantizer"
        config["quant_data"] = {"active": self.active,"wordsize": self.exponent_bits +  self.mantissa_bits + 1, "exponent_bits": self.exponent_bits,"mantissa_bits": self.mantissa_bits, "scale": self.scale}
        return config

    def quant_forward(self, inputs):
        # tf.print(self.name, "inputs", tf.shape(inputs), "self.zero_point", tf.shape(self.zero_point))
        inputs = inputs - self.zero_point
        inputs = inputs * self.scale

        # x_clip = tf.clip_by_value(inputs_recast, self.min_value, self.max_value)
        # tmp = (x_clip - self.min_value) / self.step_diff
        # q_tmp = tf.math.round(tmp)
        # y = tf.cast(q_tmp * self.step_diff + self.min_value, self.dtype)

        # prepare inputs
        sign = tf.where(inputs < 0.0, -1.0, 1.0)
        zero_inputs_mask = inputs == 0.0
        positive_inputs = inputs * sign
        positive_inputs = tf.where(zero_inputs_mask, 1.0,
                                   positive_inputs)  # replace zeros with 1.0 to mitigate math errors

        # calculate exponent for mantissa
        exponent = tf.math.log(positive_inputs) / tf.math.log(2.0)
        exponent = tf.floor(exponent)  # just floor the exponent. This exponent is used to bring the mantissa into the [1;2) range

        subnormal_mask = exponent <= -self.exponent_bias
        # normal_mask = exponent > self.exponent_bias
        exponent = tf.where(subnormal_mask, -self.exponent_bias + 1, exponent)

        # This brings the input into the range of [1;2). Basically it is x / 2^floor(ln(x))
        mantissa_float = positive_inputs / pow(2, exponent)
        # mantissa_float = tf.where(subnormal_mask,
        #                           tf.clip_by_value(mantissa_float, 0.0, 1.0),
        #                           tf.clip_by_value(mantissa_float, 1.0, 2.0))

        # now bring the mantissa from the [1;2) range into [0;1] range
        # and then into a classical fixed point representation (with a range of [0;2^n-1])
        # Like x=0.75 can be represented with step_size=0.25 -> x=0.25*3
        # Here, the rounding takes place!
        one_for_normals_only = tf.where(subnormal_mask, 0.0, 1.0)
        inted = tf.floor((mantissa_float - one_for_normals_only) * pow(2, self.mantissa_bits) + 0.5)
        # After rounding we go back baby.
        # Back to the range [1;2)
        mantissa_quanted = one_for_normals_only + (inted / pow(2, self.mantissa_bits))  # tf.round(inted) / tf.pow(2.0, m)

        # Now we have everything we need. The sign bit, the exponent (exponent) and the mantissa.
        # Well then, stitch everything together ...
        # sign in {-1, 1}   exponent in (0,2^(-bias)]    mantissa_quanted in [1;2)
        inputs_quanted = sign * pow(2, exponent) * mantissa_quanted

        # Now we are done! Just kidding!! We are not!
        # Now we have to do some kind of cleanup. Some itti bitti things to do, when you floatero...
        # Like, infinities, NaN and oh well the zeros! Do not forget the zeros!
        # First, we define some little helpers for this and that.

        inputs_quanted = self.fp_format.formatQuantizedValues(inputs_quanted, sign, exponent, mantissa_quanted,
                                                              zero_inputs_mask, inputs)
        q_min, q_max = self.fp_format.formatRange()
        clipped_values_mask = tf.logical_or(inputs_quanted == q_min, inputs_quanted == q_max)
        inputs_quanted = inputs_quanted / tf.cast(self.scale, tf.float32)
        inputs_quanted = inputs_quanted + self.zero_point
        return inputs_quanted, clipped_values_mask

    def quant(self, inputs):
        dtype = inputs.dtype
        inputs = tf.cast(inputs, tf.float32)

        @tf.custom_gradient
        # custom function, that modifies the gradient computation in the
        # backward pass
        def _quant(inputs):

            tf.cond(self.is_scaled, lambda: 0.0, lambda: tf.py_function(self.findScale, inp=[inputs], Tout=tf.float32))
            y, _ = self.quant_forward(inputs)

            # define the gradient calculation
            def grad(dy):
                # test for every element of x if it is out of the bounds of
                # the quantisation range
                is_out_of_range = tf.logical_or(inputs < self.min_value, inputs > self.max_value)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return tf.where(is_out_of_range, self.leak_clip * dy, dy)
            return y, grad
        return tf.cast(_quant(inputs), dtype=dtype)

    def getAllQuantizedValues(self):
        # All values for the mantissa as unsinged integer steps
        all_mantissa_steps = tf.range(0.0, pow(2, self.mantissa_bits))  # tf.pow(2.0, m))

        # # # Now we calculate all normal numbers
        # All normal exponents
        exponent_range = tf.range(tf.cast(0, tf.int64), powInt(2, self.exponent_bits))
        all_exponent_steps = exponent_range - tf.cast(self.exponent_bias, tf.int64) # subtract bias
        all_exponent_steps = all_exponent_steps[exponent_range != 0] # filter out the sub normal case
        all_exponent_steps = tf.cast(all_exponent_steps, tf.float32)

        # Convert the exponents to real factors
        all_exponents = pow(2, all_exponent_steps)
        # Convert the mantissas to real values
        all_mantissas = 1.0 + all_mantissa_steps / pow(2.0, self.mantissa_bits)  # 1.0 + all_mantissa_steps / tf.pow(2.0, m)

        # Multiply both together to get all representable normal numbers
        all_exponents, all_mantissas = all_exponents[None, :], all_mantissas[:, None]
        pos_values = tf.sort(tf.unique(tf.reshape(all_exponents * all_mantissas, [-1])).y)
        # print("Positive normals", tf.sort(pos_values))
        all_quantized_values = tf.concat([-pos_values, pos_values], axis=0) # (without special value representations and without subnormals)

        # # # Now we calculate all subnormal numbers
        all_exponent_subnormal_steps = tf.cast([0.0 - self.exponent_bias + 1.0], tf.float32)
        # print("subnormals exponents", tf.sort(all_exponent_subnormal_steps))
        # Convert the exponents to real factors
        all_exponents = pow(2, all_exponent_subnormal_steps)
        # print("subnormals exponents factors", tf.sort(all_exponents))
        all_mantissas = 0.0 + all_mantissa_steps / pow(2.0, self.mantissa_bits)  # 0.0 + all_mantissa_steps / tf.pow(2.0, m)
        # print("all mantissa values", all_mantissas)
        all_exponents, all_mantissas = all_exponents[None, :], all_mantissas[:, None]
        pos_values = tf.sort(tf.unique(tf.reshape(all_exponents * all_mantissas, [-1])).y)
        # print("Positive subnormals", tf.sort(pos_values))
        all_quantized_values = tf.concat([all_quantized_values, -pos_values, pos_values], axis=0)

        # print("all_exponents", all_exponents.numpy())
        # print("all_mantissas", all_mantissas.numpy())
        return tf.sort(all_quantized_values)

    def getCoeffs(self):

        coeffs = self.getAllQuantizedValues()
        coeffs = tf.sort(tf.unique(coeffs).y)# todo original was this tf.sort(tf.unique(self(coeffs)).y)
        # weired_zeros_indices = []
        # for i, coeff in enumerate(coeffs):
        #     print(coeff, "   int(coeff) == 0", int(coeff) == 0, "    coeff == int(coeff)", coeff == int(coeff))
        #     if int(coeff) == 0 and coeff == int(coeff):
        #         weired_zeros_indices.append(i)
        # if len(weired_zeros_indices) > 0: #remove duplicate zeros because of the +0 and -0 case
        #     coeffs = coeffs.tolist()
        #     for i in reversed(sorted(weired_zeros_indices)):
        #         coeffs.pop(i)
        #     coeffs = np.sort(coeffs)
        return coeffs

    def getCoeffsScaling(self):
        return pow(2, self.exponent_bias)

    def isNonUniform(self):
        return True

    def printCompleteValueRange(self):
        all_quantized_values = self.getAllQuantizedValues() # (without special value representations)
        print("all_quantized_values", all_quantized_values)

        all_quantized_values = self(all_quantized_values)
        print("all_quantized_values", all_quantized_values.numpy())
        uniq = tf.unique(all_quantized_values.numpy()).y
        print("all_quantized_values unique", uniq, "shape is", tf.shape(uniq).numpy())
        print()
