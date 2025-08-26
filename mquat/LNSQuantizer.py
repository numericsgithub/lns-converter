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
        internal_dtype (tf.dtypes.DType): Data type used internally for quantization ops.
        total_diff (float): Difference between max and min quantization points.
        step_diff (float): Difference between adjacent quantization points.
        leak_clip (float): Leak factor applied in backprop when clipping occurs.
    """

    def __init__(self, name, format="sfix", lsb=-3, msb=1, channel_wise_scaling=False, scale_inputs=False, dtype=DEFAULT_DATATYPE, internal_dtype=tf.dtypes.float32):
        # Initialize base Quantizer class with given parameters
        super().__init__(name, channel_wise_scaling=channel_wise_scaling, scale_inputs=scale_inputs, dtype=dtype)

        # Internal data type for quantization operations to reduce errors
        self.internal_dtype = internal_dtype


        # Leak factor for gradient during backpropagation when input is clipped
        self.leak_clip = 0.0
        
        # Attributes used in quant_forward (must be set properly)
        self.format = format
        self.lsb = tf.Variable(lsb, trainable=False, dtype=self.internal_dtype,
                                                    name=f"{name}_lsb")
        self.msb = tf.Variable(msb, trainable=False, dtype=self.internal_dtype,
                                                    name=f"{name}_msb")

    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend([self.lsb, self.msb])
        return variables

    def quant_forward(self, inputs):
        """
        Forward pass quantization function for sfix format only.

        Args:
            inputs (tf.Tensor): Input tensor to be quantized.

        Returns:
            tuple: (quantized tensor, boolean mask tensor indicating clipped values)
        """
        inputs = tf.cast(inputs, self.internal_dtype)

        # SFiX quantization
        log_vals = tf.math.log(tf.abs(inputs) + 1e-6) / tf.math.log(2.0)
        signs = tf.where(inputs < 0, -1.0, 1.0)



        scale_factor = 2 ** abs(self.lsb)
        q_log = tf.floor(log_vals * scale_factor + 0.5) / scale_factor

        min_val = -(2 ** (self.msb - 1))
        max_val = 2 ** (self.msb - 1) - 2 ** self.lsb

        q_log = tf.clip_by_value(q_log, min_val, max_val)




        y = tf.pow(2.0, q_log) * signs


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
            inputs_recast = tf.cast(inputs, self.internal_dtype)
            y = self.quant_forward(inputs_recast)

            def grad(dy):
                # Mask inputs outside quantization range
                is_out_of_range = tf.logical_or(inputs_recast < self.min_value, inputs_recast > self.max_value)
                # Use leak_clip factor for gradients of out-of-range inputs
                return tf.where(is_out_of_range, self.leak_clip * dy, dy)

            return y, grad

        return _quant(inputs)


