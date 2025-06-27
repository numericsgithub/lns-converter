# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import Variable
import numpy as np

from .Utilities import to_variable, to_value
from .Quantizer import Quantizer
from .QuantizerBase import DEFAULT_DATATYPE

# TODO Add variable for the self.flex_inferences stuff
# TODO Deactivate the dynamic change of the bits
# TODO Add threshold (TODO count outlier)
# TODO Implement percentile keep stuff...

class TrendPointQuantizer(Quantizer):
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

    INT_FRAC_EXTENSION = 10

    def __init__(self, name, total_bits, std_keep_factor=0.0, flex_inferences=1, leak_clip=0.0, dtype=DEFAULT_DATATYPE, debug=False, extra_flex_shift=0):
        super().__init__(name, dtype=dtype)
        self.extra_flex_shift = extra_flex_shift
        self.total_bits = to_variable(total_bits, tf.int32)
        self.flex_inferences = to_variable(flex_inferences, tf.int32)
        self.std_keep_factor = to_variable(std_keep_factor, tf.float32)
        self.debug = debug
        self.leak_clip = tf.cast(leak_clip, dtype)
        self.inference_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.input_sample = tf.Variable([], shape=[None], trainable=False, dtype=dtype, validate_shape=False)
        self.maybe_inversed_step_diff = tf.Variable(0, trainable=False, dtype=dtype)
        self.min_value = tf.Variable(0, trainable=False, dtype=dtype)
        self.max_value = tf.Variable(0, trainable=False, dtype=dtype)
        self.b_frc = tf.Variable(0, trainable=False, dtype=dtype)
        self.__error_rate_clip = tf.Variable(0, trainable=False, dtype=dtype)
        self.__error_rate_round = tf.Variable(0, trainable=False, dtype=dtype)
        self.reset()


    def reset(self):
        """
        Resets the Quantizer. The next inference triggers the coefficient search again.
        Returns:

        """
        self.inference_counter.assign(0)
        self.min_value.assign(0)
        self.max_value.assign(0)
        self.maybe_inversed_step_diff.assign(0)
        #self.input_sample.assign([])


    def collect_input_sample(self, inputs):
        tf.cond(tf.reduce_sum(tf.where(inputs > 0, 1, 1)) == 0,
                lambda :self.inference_counter.assign_add(-1),
                lambda :self.inference_counter.assign_add(0))
        self.input_sample.assign(tf.concat([tf.reshape(inputs, [-1]), self.input_sample], 0))
        return 0

    @tf.function()
    def __filterInputs(self, inputs, std_keep_factor):
        # if std_keep_factor == 0.0:
        #     return inputs, inputs
        std_keep = tf.math.reduce_std(inputs) * std_keep_factor # TODO Unique verändert das Ergebnis
        kept = tf.sort(inputs[tf.math.abs(inputs - tf.math.reduce_mean(inputs)) <= std_keep])
        kept_s = tf.size(kept)
        dropped = tf.sort(inputs[tf.math.abs(inputs - tf.math.reduce_mean(inputs)) > std_keep])
        dropped_s = tf.size(dropped)
        # if self.debug:
        #     tf.print("TrendPointQuant: ", self.name,
        #              "kept", tf.round(100 * (kept_s / tf.size(inputs))), "%", kept_s, kept,
        #              "dropped", tf.round(100 * (dropped_s / tf.size(inputs))), "%", dropped_s, dropped, "input size is", tf.size(inputs))
        clip_min = tf.reduce_min(kept)
        clip_max = tf.reduce_max(kept)
        clipped = tf.where(tf.math.abs(inputs - tf.math.reduce_mean(inputs)) <= std_keep,
                           inputs,
                           tf.clip_by_value(inputs, clip_min, clip_max))
        return clipped #kept#, clipped


    def warn_not_quantizing(self):
        tf.print("TrendPointQuant:", self.name, "is not quantizing this inference! Still creating the sample! ",
                 self.inference_counter, "/", self.flex_inferences, "inputs collected")
        return 0

    def FatalNumericalError(self):
        tf.print("TrendPointQuant:", self.name,"ERROR: TOO MANY BITS! MORE THAN 63 BITS ARE NOT SUPPORTED!")
        return 0

    def WarningNumericalError(self):
        tf.print("TrendPointQuant:", self.name,"WARNING: TOO MANY BITS! More than 24 bits are not yet tested for numerical errors!")
        return 0

    def setBitsPlot(self, stds, errors, chosen, std_errors):
        chosen = chosen.numpy()
        plt.plot(stds, errors, color="blue")
        plt.plot(stds, std_errors, color="orange")
        plt.vlines([stds[chosen]], ymin=0, ymax=np.max(errors), colors=["red"])
        plt.title(self.name)

        plt.show()
        return 0.0


    # def setBitsBeforeAndAfter(self):
    #     all_std_filters = tf.range(3.0, 22, 0.2, dtype=tf.float32)
    #     all_error, all_b_int, all_std_errors = tf.map_fn(self.setBitsBeforeAndAfterTestStd, [all_std_filters, tf.zeros_like(all_std_filters, dtype=tf.float32), tf.zeros_like(all_std_filters, dtype=tf.float32)],
    #                                     fn_output_signature=[tf.float32, tf.float32, tf.float32], parallel_iterations=1)
    #     all_error_min = tf.reduce_min(all_error)
    #     all_error_argmin = tf.argmin(all_error)
    #     #best_index = tf.reduce_max(tf.where(all_error == all_error_min))
    #     # all_error_cross = tf.where(all_std_errors <= all_error, all_error, tf.reduce_max(all_error))
    #     # all_error_cross = tf.argmin(all_error_cross)
    #     best_index = all_error_argmin
    #
    #     best_std = tf.reshape(tf.gather(all_std_filters, [best_index]), ())
    #     if self.debug:
    #         tf.print("TrendPointQuant: ", self.name, "std search got lowest error ", tf.gather(all_error, best_index), "with std filter of", tf.gather(all_std_filters, best_index))
    #     #error, best_b_int = self.setBitsBeforeAndAfterTestStd([1.0, 0.0])
    #     tf.py_function(func=self.setBitsPlot, inp=[all_std_filters, all_error/tf.cast(tf.size(self.input_sample), tf.float32),
    #                                                best_index, all_std_errors/tf.cast(tf.size(self.input_sample), tf.float32),]
    #                    , Tout=[tf.float32])
    #     self.std_keep_factor.assign(best_std)
    #     self.setBitsBeforeAndAfter_old()
    #     return 0

    # @tf.function()
    # def setBitsBeforeAndAfterTestStd(self, std_keep_factor):
    #     std_keep_factor = std_keep_factor[0]
    #     inputs = self.input_sample
    #     inputs = tf.reshape(inputs, [-1])
    #
    #     # if self.debug:
    #     #     tf.print("TrendPointQuant:", self.name, "all samples unique inputs",inputs)
    #     inputs = self.__filterInputs(inputs, std_keep_factor)
    #     #inputs = clipped
    #     if self.debug:
    #         tf.print("TrendPointQuant:", self.name, "all kept unique inputs", tf.unique(inputs)[0])
    #     int_frac_extension = tf.cast(TrendPointQuantizer.INT_FRAC_EXTENSION, dtype=tf.float64)
    #     beta = tf.cast(1.00, dtype=tf.float64)
    #
    #     t = tf.cast(self.total_bits, tf.float64)
    #     T = tf.cast(tf.range(1 - int_frac_extension, t + 1 + int_frac_extension), dtype=tf.float64)
    #     S = tf.cast(inputs, tf.float64)
    #     _05 = tf.cast(0.5, dtype=tf.float64)
    #     _1 = tf.cast(1, dtype=tf.float64)
    #     _2 = tf.cast(2, dtype=tf.float64)
    #
    #
    #     q_low = tf.where((beta * -tf.pow(_2, T - _1)) <= tf.reduce_min(S))
    #     q_low = tf.reshape(tf.gather(T, q_low), [-1])
    #     q_low = tf.concat([[t + int_frac_extension], q_low], axis=0)
    #
    #     q_high = tf.where((beta * (tf.pow(_2, T - _1) - tf.pow(_05, tf.cast(t - T, dtype=tf.float64)))) >= tf.reduce_max(S))
    #     q_high = tf.reshape(tf.gather(T, q_high), [-1])
    #     q_high = tf.concat([[t + int_frac_extension], q_high], axis=0)
    #
    #     b_int = tf.maximum(tf.reduce_min(q_low), tf.reduce_min(q_high))
    #     b_frc = t - b_int
    #     min = -tf.pow(_2, b_int - _1)
    #     max = tf.pow(_2, b_int - _1) - tf.pow(_05, t - b_int)
    #     inv_step = tf.cast(tf.pow(tf.cast(2,dtype=tf.int64),tf.cast(tf.abs(b_frc),dtype=tf.int64)),dtype=tf.float64)
    #     # self.maybe_inversed_step_diff.assign(tf.cast(inv_step, dtype=self.dtype))
    #     # self.min_value.assign(tf.cast(min, dtype=self.dtype))
    #     # self.max_value.assign(tf.cast(max, dtype=self.dtype))
    #     # self.b_frc.assign(tf.cast(b_frc, dtype=tf.float32))
    #     # self.input_sample.assign([])
    #     self_maybe_inversed_step_diff = tf.cast(inv_step, dtype=self.dtype)
    #     self_min_value = tf.cast(min, dtype=self.dtype)
    #     self_max_value = tf.cast(max, dtype=self.dtype)
    #     self_b_frc = tf.cast(b_frc, dtype=tf.float32)
    #
    #     tmp = tf.clip_by_value(self.input_sample, self_min_value, self_max_value)
    #     tmp = (tmp - self_min_value)
    #     tmp = tf.cond(self_b_frc >= 0, lambda: tmp * self_maybe_inversed_step_diff, lambda: tmp / self_maybe_inversed_step_diff)
    #     tmp = tf.floor(tmp + 0.5)
    #     tmp = tf.cond(self_b_frc >= 0, lambda: tmp / self_maybe_inversed_step_diff, lambda: tmp * self_maybe_inversed_step_diff) + self_min_value
    #     # # the check for numerical errors works by converting back the quantized number to an integer number
    #     # # quantized * step_diff == integer number
    #     # # checker = quantized * step_diff
    #     # # checker_int = int(checker)
    #     # # checker_int != checker -> numerical error!
    #     # checker = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff, lambda: tmp / self.maybe_inversed_step_diff)
    #     # checker_int = tf.cast(tf.cast(checker, tf.int64), dtype=tf.float32)
    #     # tf.print("TrendPointQuant:", self.name, "checking sample for numerical errors", inputs, tmp, self.maybe_inversed_step_diff)
    #     # tf.print("TrendPointQuant:", self.name, "checking sample for numerical errors", checker_int, checker)
    #     # tf.print("TrendPointQuant:", self.name, "checking sample for numerical errors", tf.reduce_all(tf.math.equal(checker_int, checker)))
    #     # tf.cond(tf.reduce_all(tf.math.equal(checker_int, checker)),
    #     #         lambda: 0,
    #     #         lambda: self.FatalNumericalError())
    #     # tf.cond(tf.logical_and(tf.abs(b_frc) < 64, (self.total_bits + self.INT_FRAC_EXTENSION) < 64),
    #     #         lambda: 0,
    #     #         lambda: self.FatalNumericalError())
    #     # tf.cond(tf.logical_and(tf.abs(b_frc) < 25, (self.total_bits + self.INT_FRAC_EXTENSION) < 25),
    #     #         lambda: 0,
    #     #         lambda: self.WarningNumericalError())
    #
    #     if self.debug:
    #         tf.print("TrendPointQuant: ", self.name, "setting bits based on sample size",
    #                  tf.reduce_sum(tf.where(inputs > 0, 1, 1)), "[", tf.reduce_min(inputs), ";", tf.reduce_max(inputs), "]",
    #                  " bits set to", b_int, b_frc, "[", min, ";", max, "]")
    #         tf.print("TrendPointQuant: ", self.name, "quantized sample with unique is","[", tf.reduce_min(tmp), ";", tf.reduce_max(tmp), "]", tf.unique(tmp))
    #     error = tf.cast(tf.reduce_sum(tf.pow(self.input_sample - tmp, 2)), dtype=tf.float32)
    #     std_error = tf.reduce_sum(tf.pow(self.input_sample - inputs, 2))
    #     return [error, tf.cast(b_int, dtype=tf.float32), std_error]

    @tf.function
    def setBitsBeforeAndAfter(self):
        inputs = self.input_sample
        inputs = tf.reshape(inputs, [-1])

        # if self.debug:
        #     tf.print("TrendPointQuant:", self.name, "all samples unique inputs",inputs)
        inputs = tf.cond(self.std_keep_factor != 0.0, lambda: self.__filterInputs(inputs, self.std_keep_factor), lambda: inputs)
        if self.debug:
            tf.print("TrendPointQuant:", self.name, "all kept unique inputs", tf.unique(inputs)[0])
        int_frac_extension = tf.cast(TrendPointQuantizer.INT_FRAC_EXTENSION, dtype=tf.float64)
        beta = tf.cast(1.00, dtype=tf.float64)

        t = tf.cast(self.total_bits, tf.float64)
        T = tf.cast(tf.range(1 - int_frac_extension, t + 1 + int_frac_extension), dtype=tf.float64)
        S = tf.cast(inputs, tf.float64)
        _05 = tf.cast(0.5, dtype=tf.float64)
        _1 = tf.cast(1, dtype=tf.float64)
        _2 = tf.cast(2, dtype=tf.float64)

        q_low = tf.where((beta * -tf.pow(_2, T - _1)) <= tf.reduce_min(S))
        q_low = tf.reshape(tf.gather(T, q_low), [-1])
        q_low = tf.concat([[t + int_frac_extension], q_low], axis=0)

        q_high = tf.where(
            (beta * (tf.pow(_2, T - _1) - tf.pow(_05, tf.cast(t - T, dtype=tf.float64)))) >= tf.reduce_max(S))
        q_high = tf.reshape(tf.gather(T, q_high), [-1])
        q_high = tf.concat([[t + int_frac_extension], q_high], axis=0)

        b_int = tf.maximum(tf.reduce_min(q_low), tf.reduce_min(q_high))
        b_frc = t - b_int
        min = -tf.pow(_2, b_int - _1)
        max = tf.pow(_2, b_int - _1) - tf.pow(_05, t - b_int)
        inv_step = tf.cast(tf.pow(tf.cast(2, dtype=tf.int64), tf.cast(tf.abs(b_frc), dtype=tf.int64)), dtype=tf.float64)
        self.maybe_inversed_step_diff.assign(tf.cast(inv_step, dtype=self.dtype))
        self.min_value.assign(tf.cast(min, dtype=self.dtype))
        self.max_value.assign(tf.cast(max, dtype=self.dtype))
        self.b_frc.assign(tf.cast(b_frc, dtype=tf.float32))
        self.input_sample.assign([])

        tmp = tf.clip_by_value(inputs, self.min_value, self.max_value)
        tmp = (tmp - self.min_value)
        tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff,
                      lambda: tmp / self.maybe_inversed_step_diff)
        tmp = tf.floor(tmp + 0.5)
        tmp = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff,
                      lambda: tmp * self.maybe_inversed_step_diff) + self.min_value
        # # the check for numerical errors works by converting back the quantized number to an integer number
        # # quantized * step_diff == integer number
        # # checker = quantized * step_diff
        # # checker_int = int(checker)
        # # checker_int != checker -> numerical error!
        # checker = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff, lambda: tmp / self.maybe_inversed_step_diff)
        # checker_int = tf.cast(tf.cast(checker, tf.int64), dtype=tf.float32)
        # tf.print("TrendPointQuant:", self.name, "checking sample for numerical errors", inputs, tmp, self.maybe_inversed_step_diff)
        # tf.print("TrendPointQuant:", self.name, "checking sample for numerical errors", checker_int, checker)
        # tf.print("TrendPointQuant:", self.name, "checking sample for numerical errors", tf.reduce_all(tf.math.equal(checker_int, checker)))
        # tf.cond(tf.reduce_all(tf.math.equal(checker_int, checker)),
        #         lambda: 0,
        #         lambda: self.FatalNumericalError())
        tf.cond(tf.logical_and(tf.abs(b_frc) < 64, (self.total_bits + self.INT_FRAC_EXTENSION) < 64),
                lambda: 0,
                lambda: self.FatalNumericalError())
        tf.cond(tf.logical_and(tf.abs(b_frc) < 24, (self.total_bits + self.INT_FRAC_EXTENSION) < 24),
                lambda: 0,
                lambda: self.WarningNumericalError())

        if self.debug:
            tf.print("TrendPointQuant:", self.name, "setting bits based on sample size",
                     tf.reduce_sum(tf.where(inputs > 0, 1, 1)), "[", tf.reduce_min(inputs), ";", tf.reduce_max(inputs), "]",
                     " bits set to", b_int, b_frc, "[", min, ";", max, "]")
            tf.print("TrendPointQuant:", self.name, "quantized sample with unique is", "[", tf.reduce_min(tmp), ";",
                     tf.reduce_max(tmp), "]", tf.unique(tmp))
        return 0

    def assign_add_counter(self):
        self.inference_counter.assign_add(1)
        return 0

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
        def _quant(inputs, min_value, max_value, step_diff, leak_clip): # how to remove those parameters without errors?
            self.inference_counter.assign_add(1)

            tf.cond(self.inference_counter <= self.flex_inferences and self.inference_counter >= 0,
                    lambda: self.collect_input_sample(inputs),
                    lambda: 0)
            tf.cond(self.inference_counter == self.flex_inferences,
                    lambda: self.setBitsBeforeAndAfter(),
                    lambda: 0)
            if self.debug:
                tf.cond(self.inference_counter < self.flex_inferences,
                        lambda: self.warn_not_quantizing(),
                        lambda: 0)

            tmp = tf.clip_by_value(inputs, self.min_value, self.max_value)
            tmp = (tmp - self.min_value)
            tmp = tf.cond(self.b_frc >= 0, lambda: tmp * self.maybe_inversed_step_diff, lambda: tmp / self.maybe_inversed_step_diff)
            tmp = tf.floor(tmp + 0.5)
            tmp = tf.cond(self.b_frc >= 0, lambda: tmp / self.maybe_inversed_step_diff, lambda: tmp * self.maybe_inversed_step_diff) + self.min_value

            trend_setter = tf.pow(2.0, -self.b_frc - 4.0) # self.b_frc +-???
            tf.print("TrendPointQuant:", tf.pow(2.0, -self.b_frc), trend_setter)

            q_error = inputs - tmp # 0.30 - 0.50 = -0.20 => negative trend
            tmp = tf.where(tf.logical_and(tmp != 0, q_error < 0.0),  # we do not care about the q_error == 0 case! This quantizer is not supposed to!
                           tmp - trend_setter,
                           tmp)
            tmp = tf.where(tf.logical_and(tmp != 0, q_error > 0.0),  # we do not care about the q_error == 0 case! This quantizer is not supposed to!
                           tmp + trend_setter,
                           tmp)


            # now derive the trend of a value
            # for example: q values are 0.0 0.5 1.0
            # a = 0.3 and b = 0.6 -> q(a) and q(b) = 0.5
            # but the trend gets lost there!
            # keep the negative or positive trend!
            # q(a) = 0.5 - trend_setter and q(b) + trend_setter
            # - trend_setter should be a very small number.
            #   But slow down there jonny! Do not just choose any small number!
            #   Always remember we want to work with the tf.float32 and you know how a floating point works!
            #   Don't ya mess up the mantissa! Keep this trend_setter bias close to the LSB but also not too close
            #   We do not want to mess this up! It has to be balanced!
            # - the resulting q() numbers with the trend_setter should always fall in the same pattern
            #   for example: the following q[0], q[1], q[2] values and the | showing their +/- trends
            #                are shown below. Then we use one bit less and their by loose q[1].
            #                the remaining | have exactly the same number.
            #   | q[0] |      | q[1] |      | q[2] |
            #   | q[0] |                    | q[2] |


            tmp = tf.cond(self.inference_counter < self.flex_inferences, lambda: inputs, lambda: tmp)

            # define the gradient calculation
            @tf.function
            def grad(dy):
                # test for every element of x if it is out of the bounds of
                # the quantisation range
                is_out_of_range = tf.logical_or(inputs < self.min_value, inputs > self.max_value)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return tf.where(is_out_of_range, self.leak_clip * dy, dy), tf.zeros_like(self.min_value), tf.zeros_like(
                    self.max_value), tf.zeros_like(self.maybe_inversed_step_diff), tf.zeros_like(self.leak_clip)

            return tmp, grad

        return _quant(inputs, self.min_value, self.max_value, 0.0, self.leak_clip)
