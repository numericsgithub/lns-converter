# -*- coding: utf-8 -*-

import tensorflow as tf
from mquat.Quantizer import Quantizer
from .QuantizerBase import DEFAULT_DATATYPE


class LNSQuantizer(Quantizer):
    """
    Linear quantizer class.

    This class performs linear quantization of inputs by mapping them
    to nearest points in a linear range.

    Parameters:
        name (str): Name of the TensorFlow layer.
        dtype (tf.dtypes.DType): Data type for layer operations (default float32).
        internal_dtype (tf.dtypes.DType): Data type for internal quant operations
            to reduce quantization error (default float32).
    
    Attributes:
        exponent_dtype (tf.dtypes.DType): Data type used internally for quantization ops.
        total_diff (float): Difference between max and min quantization points.
        step_diff (float): Difference between adjacent quantization points.
        leak_clip (float): Leak factor applied in backprop when clipping occurs.
    """

    def __init__(self, name, exponent_quantizer, base_is_signed=True, redefine_smallest_value_to_zero=True, epsilon=1e-9, base=2.0, dtype=DEFAULT_DATATYPE, exponent_dtype=DEFAULT_DATATYPE):
        # Initialize base Quantizer class with given parameters
        super().__init__(name, channel_wise_scaling=False, scale_inputs=False, dtype=dtype)

        self.base = base
        self.base_is_signed = base_is_signed
        self.redefine_smallest_value_to_zero = redefine_smallest_value_to_zero
        self.epsilon = epsilon
        self.exponent_is_signed = tf.Variable(initial_value=-1.0, trainable=False, name=name+"_exponent_is_signed")

        # Internal data type for quantization operations to reduce errors
        self.exponent_dtype = exponent_dtype

        # Leak factor for gradient during backpropagation when input is clipped
        self.leak_clip = 0.0
        
        # Attributes used in quant_forward (must be set properly)
        self.exponent_quantizer = exponent_quantizer

    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = [self.exponent_is_signed]
        variables.extend(self.exponent_quantizer.getQuantVariables())
        return variables

    def quant_debug(self, inputs):
        inputs = tf.cast(inputs, self.exponent_dtype)

        # SFiX quantization
        log_vals = tf.math.log(tf.abs(inputs) + self.epsilon) / tf.math.log(self.base)

        # Flip signs (So if everything is negative we handle it as everything is positive so we can ignore the sign properly)
        # So this flipping is done because of internal reasons. The FlexPointQuantizer will use a ufix instead of sfix if everything is positive.
        # So this step allows the FlexPointQuantizer to use a ufix if everything is negative.
        q_log = tf.cond(self.exponent_is_signed == 1.0, lambda: self.exponent_quantizer(log_vals), lambda: -self.exponent_quantizer(-log_vals))

        return q_log, log_vals

    def define_exponent_sign(self, log_vals):
        self.exponent_is_signed.assign(tf.cond(tf.reduce_max(log_vals) <= 0.0, lambda: 0.0, lambda: 1.0))
        return 1.0


    def quant_forward(self, inputs):
        """
        Forward pass quantization function for sfix format only.

        Args:
            inputs (tf.Tensor): Input tensor to be quantized.

        Returns:
            tuple: (quantized tensor, boolean mask tensor indicating clipped values)
        """
        inputs = tf.cast(inputs, self.exponent_dtype)

        # Get the exponent in full precision. So we then have: self.base^log_vals
        # We only do this with the absolute values. This way, the log function only gets positive values.
        # The log of a negative value is undefined!
        log_vals = tf.math.log(tf.abs(inputs) + self.epsilon) / tf.math.log(self.base)

        # If the base is signed, we will handle negative numbers too!
        signs = tf.where(inputs < 0, -1.0, 1.0)

        # Now we quantize the full precision exponent (log_vals) with the exponent_quantizer!
        # That's pretty much it. But, we have to do a small trick:
        # Flip signs (So if everything is negative we handle it as everything is positive so we can ignore the sign properly)
        # So this flipping is done because of internal reasons. The FlexPointQuantizer will use a ufix instead of sfix if everything is positive.
        # So this step allows the FlexPointQuantizer to use a ufix if everything is negative.
        tf.cond(self.exponent_is_signed == -1.0, lambda: self.define_exponent_sign(log_vals), lambda: 0.0)
        q_log = tf.cond(self.exponent_is_signed == 1.0, lambda: self.exponent_quantizer(log_vals), lambda: -self.exponent_quantizer(-log_vals))

        # Ok, now we create a mask to mark all numbers that, if redefine_smallest_value_to_zero is set, should be defined as zero instead.
        # This is the Trick shown in "Low-precision logarithmic arithmetic  for neural network accelerators" page 76, section E. "Encoding of zeroes"
        # We take the biggest negative exponent value and define it as zero.
        zero_mask = tf.cond(self.exponent_is_signed == 1.0, lambda: q_log == self.exponent_quantizer.min_value, lambda: -q_log == self.exponent_quantizer.max_value)


        # Now we calculate the linear value with the quantized exponents: self.base^q_log
        y = tf.pow(self.base, q_log)
        if self.base_is_signed:
            y = y * signs
        else:
            y = tf.where(signs == -1.0, 0.0, y)

        if self.redefine_smallest_value_to_zero:
            y = tf.where(zero_mask, 0.0, y)

        return tf.cast(y, self.dtype)


    def quant(self, inputs):
        """
        Quantization function with custom gradient.

        Applies quantization during forward pass and modifies gradient
        computation to use leak factor when inputs are clipped.

        Args:
            inputs (tf.Tensor): Input tensor(s).

        Returns:
            tf.Tensor: Quantized output tensor.
        """
        @tf.custom_gradient
        def _quant(inputs):
            inputs_recast = tf.cast(inputs, self.exponent_dtype)
            y = self.quant_forward(inputs_recast)

            def grad(dy):
                # Mask inputs outside quantization range
                is_out_of_range = tf.logical_or(inputs_recast < self.min_value, inputs_recast > self.max_value)
                # Use leak_clip factor for gradients of out-of-range inputs
                return tf.where(is_out_of_range, self.leak_clip * dy, dy)

            return y, grad

        return _quant(inputs)


