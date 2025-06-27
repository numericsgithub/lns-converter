# -*- coding: utf-8 -*-

import tensorflow as tf
from .Quantizer import Quantizer
from .QuantizerBase import DEFAULT_DATATYPE


class AsymmetricQuantizer(Quantizer):
    """linear quantisizer

    the linear quantisizer quantisizes the input to the nearest points specified
    in a linear range.

    Parameters:
        name (string):
            the name of the layer in the TensorFlow graph.
        dtype (tf.dtypes.DType):
            datatype of the layers's operations.
            default is float32.
        internal_dtype (tf.dtypes.DType):
            internal datatype of the layers's quantization operations.
            prevent quantization errors(wrong rounding) if quantization range is not exactly representable by float32.
            default is float64.
    Attributes:
        internal_dtype (tf.dtypes.DType):
            internal datatype of the layers's quantization operations.
            prevent quantization errors(wrong rounding) if quantization range is not exactly representable by float32.
            default is float64.
        total_diff (float):
            difference between the min and the max quantisation point.
        step_diff (float):
            difference between two neighbouring quantization points.
        leak_clip (float):
            leak factor for backpropagation.
            applied when values are clipped to quantsiation range.
    """

    def __init__(self, name, total_bits, sigma, min_value=0.0, max_value=0.0, dtype=DEFAULT_DATATYPE, internal_dtype=tf.dtypes.float64):
        super().__init__(name, dtype)
        self.internal_dtype = internal_dtype
        self.leak_clip = 0.0
        self.total_bits = total_bits
        self.min_value = tf.Variable(min_value, trainable=False, dtype=tf.float32)
        self.max_value = tf.Variable(max_value, trainable=False, dtype=tf.float32)
        self.sigma = sigma
        self.is_signed = False # all is input right now. implement it later
        self.is_sigma = True # well it will be like this for now...
        self.percent = self.sigma

    def _sigma(self, tensor):
        if not self.is_signed:
            return tf.math.reduce_std(tensor[tensor > 0])
        return tf.math.reduce_std(tensor)

    def firstInit(self, inputs):
        self.min_value.assign(0.0) # all is input right now
        x_max = tf.reduce_max(inputs)
        alpha = self.percent * tf.reduce_max(tf.abs(inputs))
        if self.is_sigma:
            cur_sigma = self._sigma(inputs)
            alpha = self.percent * cur_sigma
            if self.is_signed:
                # We also consider the signed activation. Other framworks will skip this tensor.
                alpha = self.percent * cur_sigma / 1.25

            # For a higher bit-width, using a wider range still will not cause accuracy loss.
            if self.total_bits < 6:
                # For small bit, need clip.
                alpha = tf.minimum(alpha, x_max)
        self.max_value.assign(alpha)
        return 0.0


    def linear_dequantize(self, input, scale, zero_point):
        """
        Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
        input: integer input tensor to be mapped
        scale: scaling factor for quantization
        zero_pint: shift for quantization
        """

        # reshape scale and zeropoint for convolutional weights and activation
        # if len(input.shape) == 4:
        #     scale = scale.view(-1, 1, 1, 1)
        #     zero_point = zero_point.view(-1, 1, 1, 1)
        # # reshape scale and zeropoint for linear weights
        # elif len(input.shape) == 2:
        #     scale = scale.view(-1, 1)
        #     zero_point = zero_point.view(-1, 1)
        # mapping integer input to fixed point float point value with given scaling factor and zeropoint
        # if inplace:
        #     input.add_(zero_point).div_(scale)
        #     return input
        return (input + zero_point) / scale


    def linear_dequantize2(self, input):
        """
        Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
        input: integer input tensor to be mapped
        scale: scaling factor for quantization
        zero_pint: shift for quantization
        """
        n = 2 ** self.total_bits - 1
        scale = n / tf.maximum((self.max_value - self.min_value),
                               1e-8)  # n / torch.clamp((saturation_max - saturation_min), min=1e-8)
        zero_point = scale * self.min_value

        # if integral_zero_point:
        zero_point = tf.floor(zero_point + 0.5)
        # if signed:
        zero_point += 2 ** (self.total_bits - 1)
        return input / 128.0


    def linear_quantize(self, input, scale, zero_point):
        """
        Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
        input: single-precision input tensor to be quantized
        scale: scaling factor for quantization
        zero_point: shift for quantization
        """

        # # reshape scale and zeropoint for convolutional weights and activation
        # if len(input.shape) == 4:
        #     scale = scale.view(-1, 1, 1, 1)
        #     zero_point = zero_point.view(-1, 1, 1, 1)
        # # reshape scale and zeropoint for linear weights
        # elif len(input.shape) == 2:
        #     scale = scale.view(-1, 1)
        #     zero_point = zero_point.view(-1, 1)
        # # mapping single-precision input to integer values with the given scale and zeropoint
        # if inplace:
        #     input.mul_(scale).sub_(zero_point).round_()
        #     return input
        return scale * input - zero_point


    def linear_quantize2(self, input):
        """
        Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
        input: single-precision input tensor to be quantized
        scale: scaling factor for quantization
        zero_point: shift for quantization
        """
        # n = 2 ** self.total_bits - 1
        # scale = n / tf.maximum((self.max_value - self.min_value),
        #                        1e-8)  # n / torch.clamp((saturation_max - saturation_min), min=1e-8)
        # zero_point = scale * self.min_value
        #
        # # if integral_zero_point:
        # zero_point = tf.floor(zero_point + 0.5)
        # # if signed:
        # zero_point += 2 ** (self.total_bits - 1)
        return input * 128.0 # scale * input - zero_point


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
            # tf.cond(tf.logical_and(self.min_value == 0.0, self.max_value == 0.0),
            #         lambda: self.firstInit(inputs), lambda : 0.0) # todo Do not comment this out. but maybe it is wrong??

            n = 2 ** self.total_bits - 1
            scale = n / tf.maximum((self.max_value - self.min_value),
                                 1e-8)  # n / torch.clamp((saturation_max - saturation_min), min=1e-8)
            zero_point = scale * self.min_value

            # if integral_zero_point:
            zero_point = tf.floor(zero_point + 0.5)
            # if signed:
            zero_point += 2 ** (self.total_bits - 1)

            # inputs_recast = tf.cast(inputs, self.internal_dtype)
            # x_clip = tf.clip_by_value(inputs_recast, self.min_value, self.max_value)
            # tmp = (x_clip - self.min_value) / self.step_diff
            # q_tmp = tf.math.round(tmp)
            # y = tf.cast(q_tmp * self.step_diff + self.min_value, self.dtype)
            new_quant_x = tf.floor(self.linear_quantize(inputs, scale, zero_point) + 0.5)
            n = 2 ** (self.total_bits - 1)
            new_quant_x = tf.clip_by_value(new_quant_x, -n, n - 1.0)
            quant_x = self.linear_dequantize(new_quant_x,
                                        scale,
                                        zero_point)
            y = quant_x

            # define the gradient calculation
            def grad(dy):
                # test for every element of x if it is out of the bounds of
                # the quantisation range
                is_out_of_range = tf.logical_or(inputs < self.min_value, inputs > self.max_value)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return tf.where(is_out_of_range, self.leak_clip * dy, dy)

            return y, grad

        return _quant(inputs)
