# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import Variable
import numpy as np

#from . import Logging
from .Utilities import to_variable, to_value
from .Quantizer import Quantizer
import random
import time
from .QuantizerBase import DEFAULT_DATATYPE

# TODO Add variable for the self.flex_inferences stuff
# TODO Deactivate the dynamic change of the bits
# TODO Add threshold (TODO count outlier)
# TODO Implement percentile keep stuff...

class FlexPointQuantizer(Quantizer):
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

    INT_FRAC_EXTENSION = 14
    PLOTTING = False

    def __init__(self, name, total_bits, leak_clip=0.0, dtype=DEFAULT_DATATYPE, debug=False, extra_flex_shift=0, set_b_int=None, is_symmetric=False, pre_filters=[], post_filters=[], round_to_nearest=True, hirachy_value=None):
        super().__init__(name, dtype=dtype, pre_filters=pre_filters, post_filters=post_filters, hirachy_value=hirachy_value)
        self.extra_flex_shift = extra_flex_shift
        self.total_bits = to_variable(total_bits, tf.int32)
        self.debug = debug
        self.leak_clip = tf.cast(leak_clip, dtype)
        self.round_to_nearest = round_to_nearest
        self.maybe_inversed_step_diff = tf.Variable(0, name=self.name+"_maybe_inversed_step_diff", trainable=False, dtype=tf.float32)
        self.min_value = tf.Variable(0, name=self.name+"_min_value", trainable=False, dtype=dtype)
        self.max_value = tf.Variable(0, name=self.name+"_max_value", trainable=False, dtype=dtype)
        self.best_std_filter = tf.Variable(0, name=self.name+"_std_filter", trainable=False, dtype=dtype)
        self.b_frc = tf.Variable(0, name=self.name+"_b_frc", trainable=False, dtype=dtype)
        self.b_int_override = set_b_int
        self.is_symmetric = is_symmetric
        if set_b_int is not None:
            self.b_int_override = tf.constant(self.b_int_override, dtype=tf.float64)

    def get_sfix_settings(self):
        lsb = self.b_frc.numpy() * -1
        msbs = int(lsb + self.total_bits.numpy())
        if self.min_value == 0:
            msbs += 1
        return msbs, int(lsb)

    @property
    def doc_string(self):
        doc = ""
        doc += "min: " + str(float(self.min_value.numpy())).ljust(10)
        doc += "max: " + str(float(self.max_value.numpy())).ljust(10)
        doc += "b_frc: " + str(float(self.b_frc.numpy())).ljust(10)
        doc += "total_bits: " + str(float(self.total_bits.numpy())).ljust(10)
        return doc

    def reset(self):
        """
        Resets the Quantizer. The next inference triggers the coefficient search again.
        Returns:

        """
        self.maybe_inversed_step_diff.assign(0)
        self.min_value.assign(0)
        self.max_value.assign(0)
        self.b_frc.assign(0)
        # self.setBitsBeforeAndAfter(tf.constant([-8, -4, -2, -1, 0, 1, 2, 4, 8], dtype=tf.float32) / 32.0)
        #self.b_frc_trend_setter.assign(0)


    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend([self.maybe_inversed_step_diff, self.min_value, self.max_value,
                          self.b_frc])
        return variables

    def getCoeffs(self):
        step_size = tf.cond(self.b_frc >= 0, lambda: 1.0 / self.maybe_inversed_step_diff, lambda: 1.0 * self.maybe_inversed_step_diff)
        coeffs = tf.range(tf.cast(self.min_value, tf.float32), tf.cast(self.max_value + step_size, tf.float32), tf.cast(step_size, tf.float32))
        # tf.print("coeffs asd", coeffs, self.min_value, self.max_value, step_size)
        return coeffs

    def getCoeffsScaling(self):
        return (1.0 / tf.reduce_max(tf.abs(self.getCoeffs()))) * tf.pow(2.0, float(self.total_bits) - 1.0)

    def isNonUniform(self):
        return False

    def set_b_frac(self, b_frc):
        t = tf.cast(self.total_bits, tf.float64)
        _05 = tf.cast(0.5, dtype=tf.float64)
        _1 = tf.cast(1, dtype=tf.float64)
        _2 = tf.cast(2, dtype=tf.float64)

        b_int = t - b_frc
        min = -tf.pow(_2, b_int - _1)
        if not self.is_symmetric:
            max = tf.pow(_2, b_int - _1) - tf.pow(_05, t - b_int)
        else:
            max = -min
        inv_step = tf.cast(tf.pow(tf.cast(2, dtype=tf.int64), tf.cast(tf.abs(b_frc), dtype=tf.int64)), dtype=tf.float64)
        self.maybe_inversed_step_diff.assign(tf.cast(inv_step, dtype=tf.float32))
        self.min_value.assign(tf.cast(min, dtype=self.dtype))
        self.max_value.assign(tf.cast(max, dtype=self.dtype))
        self.b_frc.assign(tf.cast(b_frc, dtype=DEFAULT_DATATYPE))

    @tf.function(jit_compile=True)
    def filterInputs(self, inputs, std_keep_factor):
        if std_keep_factor == 0.0:
            return inputs
        std_keep = tf.math.reduce_std(inputs) * std_keep_factor
        kept = inputs[tf.math.abs(inputs - tf.math.reduce_mean(inputs)) <= std_keep]
        # kept_s = tf.size(kept)
        # dropped = tf.sort(inputs[tf.math.abs(inputs - tf.math.reduce_mean(inputs)) > std_keep])
        # dropped_s = tf.size(dropped)
        # if self.debug:
        #     tf.print("FlexPointQuant: ", self.name,
        #              "kept", tf.round(100 * (kept_s / tf.size(inputs))), "%", kept_s, kept,
        #              "dropped", tf.round(100 * (dropped_s / tf.size(inputs))), "%", dropped_s, dropped, "input size is", tf.size(inputs))
        clip_min = tf.reduce_min(kept)
        clip_max = tf.reduce_max(kept)
        clipped = tf.clip_by_value(inputs, clip_min, clip_max)#tf.where(tf.math.abs(inputs - tf.math.reduce_mean(inputs)) <= std_keep,
                  #         inputs,
                  #         tf.clip_by_value(inputs, clip_min, clip_max))
        return clipped #kept#, clipped

    @tf.function(autograph=False)
    def setBitsBeforeAndAfter(self, input_sample):
        inputs = input_sample
        inputs = tf.cond(tf.reduce_all(inputs == 0.0), lambda : tf.cast([-1.0/64, 1.0/64], inputs.dtype), lambda : inputs)
        inputs = tf.reshape(inputs, [-1])
        tf.print(self.name, "INPUTS SAMPLE", tf.reduce_min(inputs), tf.reduce_max(inputs), tf.size(tf.unique(inputs).y), tf.unique(inputs).y)

        int_frac_extension = tf.cast(FlexPointQuantizer.INT_FRAC_EXTENSION, dtype=tf.float64)

        t = tf.cast(self.total_bits, tf.float64)
        T = tf.cast(tf.range(1 - int_frac_extension - t, 1 + int_frac_extension), dtype=tf.float64)
        S = tf.cast(inputs, tf.float64)
        _05 = tf.cast(0.5, dtype=tf.float64)
        _1 = tf.cast(1, dtype=tf.float64)
        _2 = tf.cast(2, dtype=tf.float64)

        if True: #self.name.endswith("_out"):
            @tf.function(jit_compile=True)
            def quantize(min, max, step, inputs):
                max = tf.cond(tf.reduce_all(tf.abs(inputs) == inputs), lambda :max + tf.abs(min), lambda :max)
                min = tf.cond(tf.reduce_all(tf.abs(inputs) == inputs), lambda :tf.cast(0.0, min.dtype), lambda :min)
                inputs = tf.clip_by_value(inputs, min, max)
                tmp = inputs / step
                if self.round_to_nearest:
                    tmp = tf.floor(tmp + 0.5)
                else:
                    tmp = tf.floor(tmp + 0.0)
                    #tmp = tf.floor(tmp + 0.0)
                tmp = tmp * step
                return tmp

            q_low = -tf.pow(_2, T - _1)
            q_high = tf.pow(_2, T - _1) - tf.pow(_05, tf.cast(t - T, dtype=tf.float64))
            q_step = tf.pow(_2, T - t)
            q_step_abs = tf.pow(_2, tf.abs(T - t))
            use_std_filter = tf.repeat(True if not self.name.endswith("_out") else False, tf.size(q_step))
            #use_std_filter = tf.repeat(True, tf.size(q_step))

            def test_test_it(x):
                l, h, step, std_filter = x
                f_inputs = self.filterInputs(tf.cast(inputs, DEFAULT_DATATYPE), tf.cast(std_filter, DEFAULT_DATATYPE))
                result = quantize(tf.cast(l, DEFAULT_DATATYPE) , tf.cast(h, DEFAULT_DATATYPE), tf.cast(step, DEFAULT_DATATYPE), tf.cast(f_inputs, DEFAULT_DATATYPE))
                result = tf.cast(result, tf.float32)
                sad = tf.cast(-tf.reduce_sum(tf.abs(result - tf.cast(inputs, tf.float32))), tf.float64)
                verity = tf.cast(tf.size(tf.unique(result).y), tf.float64)
                mse_sad = tf.cast(tf.reduce_sum(tf.pow(tf.cast(inputs, tf.float32) - result, 2)), tf.float64)
                return mse_sad, verity

            def test_it(x):
                use_filter, l, h, step = x
                all_std_filters = tf.range(3.0, 8, 0.25, dtype=tf.float32)
                all_std_filters = tf.cond(use_filter, lambda : all_std_filters, lambda : tf.ones_like(all_std_filters) * 99.0 )
                l = tf.repeat(l, tf.size(all_std_filters))
                h = tf.repeat(h, tf.size(all_std_filters))
                step = tf.repeat(step, tf.size(all_std_filters))

                all_mse_sad, all_verity = tf.map_fn(test_test_it, elems=[l, h, step, all_std_filters],
                                        fn_output_signature=(tf.float64, tf.float64), back_prop=False,
                                        parallel_iterations=1)
                mse_sad_argmin = tf.argmin(all_mse_sad)
                best_sad = tf.reduce_min(all_mse_sad)
                best_std = tf.reshape(tf.gather(all_std_filters, [mse_sad_argmin]), ())
                best_verity = tf.reshape(tf.gather(all_verity, [mse_sad_argmin]), ())
                return best_sad, best_verity, tf.cast(best_std, tf.float64)

            sad, verity, std_filter = tf.map_fn(test_it, elems=[
                use_std_filter,
                tf.cast(q_low, tf.float32),
                tf.cast(q_high, tf.float32),
                tf.cast(q_step, tf.float32)],
                fn_output_signature=(tf.float64, tf.float64, tf.float64), back_prop=False, parallel_iterations=1)

            # best_indexes = tf.where(verity == tf.reduce_max(verity))
            # sad_vals_for_best = tf.gather(sad, best_indexes)
            # best_index_overall = tf.gather(best_indexes, tf.argmax(sad_vals_for_best))
            # best_index_overall = tf.cast(tf.reduce_max(tf.reshape(best_index_overall, [-1])), tf.int64)
            # # tf.print("TEST best_indexes", best_indexes)
            # # tf.print("TEST sad_vals_for_best", sad_vals_for_best)
            # # tf.print("TEST best_index_overall", best_index_overall)

            # result = verity
            tf.print(self.name, "SEARCH ver RESULT", verity, summarize=-1)
            tf.print(self.name, "SEARCH MIN RESULT", q_low, summarize=-1)
            tf.print(self.name, "SEARCH MAX RESULT", q_high, summarize=-1)
            tf.print(self.name, "SEARCH STEP RESULT", q_step, summarize=-1)
            tf.print(self.name, "SEARCH sad RESULT", sad, summarize=-1)
            tf.print(self.name, "SEARCH std_filter RESULT", std_filter, summarize=-1)

            # best_value = tf.reduce_max(result)
            # best_last_index = tf.cast(tf.argmax(result), tf.int32) - 1 + tf.reduce_sum(tf.where(result == best_value, 1, 0))
            #result = tf.reshape(result, [tf.size(result)])
            #results_rev = tf.reverse(result, axis=0)

            if not self.name.endswith("_out"):
                result = sad
                # max_field = tf.cast(tf.floor((tf.reduce_sum(tf.where(tf.reduce_max(result) == result, 1.0, 0.0)) - 1.0) / 2.0 + 0.5), tf.int64)
                # max_field = tf.minimum(max_field, 1)
                best_last_index = tf.cast(tf.shape(result)[0], tf.int64) - tf.argmin(tf.reverse(result, [0]), axis=0)-1 #+ max_field
            else:
                result = sad
                best_last_index = tf.cast(tf.shape(result)[0], tf.int64) - tf.argmin(tf.reverse(result, [0]), axis=0)-1 #+ max_field
                best_last_index = tf.minimum(tf.cast(tf.size(result) - 1, tf.int64), best_last_index)
                # result = verity
                # max_field = tf.cast(tf.floor((tf.reduce_sum(tf.where(tf.reduce_max(result) == result, 1.0, 0.0)) - 1.0) / 2.0 + 0.5), tf.int64)
                # max_field = tf.minimum(max_field, 1)
                # best_last_index = tf.cast(tf.shape(result)[0] - tf.argmax(tf.reverse(result, [0]), axis=0)-1 + 1, tf.int64) #+ max_field
            #best_last_index = tf.cond(tf.all(result == 1), lambda : tf.cast(1 + int_frac_extension + t, tf.int64), lambda : best_last_index)
            best_std_filter = tf.reduce_max(tf.gather(std_filter, best_last_index)) # tf.reduce_max is only there to enforce a scalar and not [1] shape
            tf.print(self.name, q_high)
            tf.print(self.name, "CHOSEN INDEX WAS, ", best_last_index,
                     "sad:", tf.reshape(tf.gather(sad, best_last_index), [-1]),
                     "verity:", tf.reshape(tf.gather(verity, best_last_index), [-1]),
                     "std_filter:", best_std_filter,
                     "q_low:", tf.reshape(tf.gather(q_low, best_last_index), [-1]),
                     "q_high:", tf.reshape(tf.gather(q_high, best_last_index), [-1]),
                     "q_step:", tf.reshape(tf.gather(q_step, best_last_index), [-1]))

            T_best = tf.cast(1 - int_frac_extension - t + tf.cast(best_last_index, tf.float64), dtype=tf.float64)
            best_l = -tf.pow(_2, T_best - _1)
            best_h = tf.pow(_2, T_best - _1) - tf.pow(_05, tf.cast(t - T_best, dtype=tf.float64))
            q_step_best = tf.pow(_2, T_best - t)
            best_abs_step = tf.pow(_2, tf.abs(T_best - t))
            best_t = t - T_best

            best_h = tf.cond(tf.reduce_all(tf.abs(inputs) == inputs), lambda :best_h + tf.abs(best_l), lambda :best_h)
            best_l = tf.cond(tf.reduce_all(tf.abs(inputs) == inputs), lambda :tf.cast(0.0, best_l.dtype), lambda :best_l)


            self.maybe_inversed_step_diff.assign(tf.cast(best_abs_step, dtype=tf.float32))
            self.min_value.assign(tf.cast(best_l, dtype=self.dtype))
            self.max_value.assign(tf.cast(best_h, dtype=self.dtype))
            self.best_std_filter.assign(tf.cast(best_std_filter, dtype=DEFAULT_DATATYPE))
            self.b_frc.assign(tf.cast(best_t, dtype=DEFAULT_DATATYPE))

        else:
            q_low = tf.where(( -tf.pow(_2, T - _1)) <= tf.reduce_min(S))
            q_low = tf.reshape(tf.gather(T, q_low), [-1])
            q_low = tf.concat([[t + int_frac_extension], q_low], axis=0)

            q_high = tf.where(
                ( (tf.pow(_2, T - _1) - tf.pow(_05, tf.cast(t - T, dtype=tf.float64)))) >= tf.reduce_max(S))
            q_high = tf.reshape(tf.gather(T, q_high), [-1])
            q_high = tf.concat([[t + int_frac_extension], q_high], axis=0)

            b_int = tf.maximum(tf.reduce_min(q_low), tf.reduce_min(q_high))
            if self.b_int_override is not None:
                b_int = self.b_int_override
            b_frc = t - b_int
            min = -tf.pow(_2, b_int - _1)
            if not self.is_symmetric:
                max = tf.pow(_2, b_int - _1) - tf.pow(_05, t - b_int)
            else:
                max = -min
            inv_step = tf.cast(tf.pow(tf.cast(2, dtype=tf.int64), tf.cast(tf.abs(b_frc), dtype=tf.int64)), dtype=tf.float64)
            self.maybe_inversed_step_diff.assign(tf.cast(inv_step, dtype=tf.float32))
            self.min_value.assign(tf.cast(min, dtype=self.dtype))
            self.max_value.assign(tf.cast(max, dtype=self.dtype))
            self.b_frc.assign(tf.cast(b_frc, dtype=DEFAULT_DATATYPE))

        # tmp = tf.clip_by_value(inputs, self.min_value, self.max_value)
        # tmp = (tmp - self.min_value)
        # tmp = tf.cond(self.b_frc >= 0, lambda: tf.cast(tmp, tf.float32) * self.maybe_inversed_step_diff,
        #               lambda: tf.cast(tmp, tf.float32) / self.maybe_inversed_step_diff)
        # tmp = tf.floor(tmp + 0.5)
        # tmp = tf.cond(self.b_frc >= 0, lambda: tf.cast(tmp, tf.float32) / self.maybe_inversed_step_diff,
        #               lambda: tf.cast(tmp, tf.float32) * self.maybe_inversed_step_diff) + tf.cast(self.min_value, tf.float32)
        # tmp = tf.cast(tmp, DEFAULT_DATATYPE)


        test_result = self.quant_forward(self.filterInputs(inputs, self.best_std_filter))
        tf.print(self.name, "TEST RESULT IS", tf.reduce_min(test_result), tf.reduce_max(test_result), tf.size(tf.unique(test_result).y))
        tf.print(self.name, "TEST RESULT IS", self.min_value, self.max_value)
        # tf.cond(tf.logical_and(tf.abs(b_frc) < 64, (self.total_bits + self.INT_FRAC_EXTENSION) < 64),
        #         lambda: 0,
        #         lambda: self.FatalNumericalError())
        # tf.cond(tf.logical_and(tf.abs(b_frc) < 24, (self.total_bits + self.INT_FRAC_EXTENSION) < 24),
        #         lambda: 0,
        #         lambda: self.WarningNumericalError())

        # direct_hit = tf.reduce_sum(tf.where(inputs == tmp, 1, 0))
        # # Logging.Log(self.name + "_quant_inputs", tmp)
        # # Logging.Log(self.name + "_direct_hits", direct_hit)
        # sample_size = tf.reduce_sum(tf.where(inputs > 0, 1, 1))
        # tf.print("FlexPointQuant:", self.name, "got", sample_size, "weights with", direct_hit, "direct matches (",(direct_hit/sample_size)*100.0,"%)", "[", min, ";", max, "]")

        #FlexPointQuantizer.error_statistic.assign(tf.concat([tf.reshape(inputs - tmp, [-1]), FlexPointQuantizer.error_statistic], 0))

        # if self.debug:
        #     tf.print("FlexPointQuant:", self.name, "setting bits based on sample size",
        #              tf.reduce_sum(tf.where(inputs > 0, 1, 1)), "[", tf.reduce_min(inputs), ";", tf.reduce_max(inputs), "]",
        #              " bits set to", b_int, b_frc, "[", min, ";", max, "]")
        #     tf.print("FlexPointQuant:", self.name, "quantized sample with unique is", "[", tf.reduce_min(tmp), ";",
        #              tf.reduce_max(tmp), "]", tf.unique(tmp))
        return 0

    def to_sfix(self, inputs):
        tmp = tf.clip_by_value(inputs, self.min_value, self.max_value)
        tmp = tf.cond(self.b_frc >= 0, lambda: tf.cast(tmp, tf.float32) * self.maybe_inversed_step_diff,
                      lambda: tf.cast(tmp, tf.float32) / self.maybe_inversed_step_diff)
        rounded = tf.floor(tmp + 0.5)
        return rounded

    @tf.function(jit_compile=True)
    def quant_forward(self, inputs):
        ## tf.print(self.name, "TESTSETST1", tf.size(tf.unique(tf.reshape(inputs, [-1])).y), tf.unique(tf.reshape(inputs, [-1])).y)
        epsilon = tf.cast(0.001, DEFAULT_DATATYPE)

        #inputs = self.filterInputs(inputs, self.best_std_filter)

        tmp = tf.clip_by_value(inputs, self.min_value, self.max_value)
        ## tf.print(self.name, "TESTSETST2", tf.size(tf.unique(tf.reshape(tmp, [-1])).y), tf.unique(tf.reshape(tmp, [-1])).y)
        tmp = (tmp - self.min_value)
        ## tf.print(self.name, "TESTSETST3", tf.size(tf.unique(tf.reshape(tmp, [-1])).y), tf.unique(tf.reshape(tmp, [-1])).y)
        ## tf.print(self.name, "self.maybe_inversed_step_diff", self.maybe_inversed_step_diff, self.min_value, self.max_value)
        tmp = tf.cond(self.b_frc >= 0, lambda: tf.cast(tmp, tf.float32) * self.maybe_inversed_step_diff,
                      lambda: tf.cast(tmp, tf.float32) / self.maybe_inversed_step_diff)
        if self.round_to_nearest:
            rounded = tf.floor(tmp + 0.5)
        else:
            rounded = tf.floor(tmp + 0.0)

        rounded = tf.cond(self.b_frc >= 0, lambda: rounded / self.maybe_inversed_step_diff,
                          lambda: rounded * self.maybe_inversed_step_diff) + tf.cast(self.min_value, tf.float32)
        ## tf.print(self.name, "TESTSETST6", tf.size(tf.unique(tf.reshape(rounded, [-1])).y), tf.unique(tf.reshape(rounded, [-1])).y)
        rounded = tf.cast(rounded, DEFAULT_DATATYPE)
        return rounded

    def warn_about_collapse(self, unique_unquantized_values, unique_quantized_values, inputs):
        tf.print(self.name, "CRITICAL! INPUTS ARE QUANTIZED TO THE SAME VALUE:", unique_quantized_values, "INPUTS WERE:", unique_unquantized_values)
        self.reset()
        self.setBitsBeforeAndAfter(inputs)
        return 0.0

    def quant(self, org_inputs):
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
        def _quant(inputs, min_value, max_value, leak_clip): # how to remove those parameters without errors?
            tmp = inputs
            # for pre_filter in self.pre_filters:
            #     tmp = pre_filter(tmp, inputs)
            tf.cond(min_value == max_value,
                    lambda: self.setBitsBeforeAndAfter(tmp),
                    lambda: 0)

            y = self.quant_forward(tmp)

            # inputs_unique = tf.unique(tf.reshape(inputs, [-1])).y
            # y_unique = tf.unique(tf.reshape(y, [-1])).y
            #
            # y_size = tf.cast(tf.size(y_unique), tf.float32)
            # tf.cond(y_size <= 2.0,
            #         lambda: self.warn_about_collapse(inputs_unique, y_unique, inputs),
            #         lambda: 0.0)

            # for post_filters in self.post_filters:
            #     tmp = post_filters(tmp, inputs)

            # define the gradient calculation
            def grad(dy): # , variables=None
                # test for every element of a if it is out of the bounds of
                # the quantisation range
                is_out_of_range = tf.logical_or(inputs < min_value, inputs > max_value)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return tf.where(is_out_of_range, leak_clip * dy, dy), tf.zeros_like(min_value), tf.zeros_like(max_value), tf.zeros_like(leak_clip)

            return y, grad
        return _quant(org_inputs, self.min_value, self.max_value, self.leak_clip)
