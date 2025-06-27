# -*- coding: utf-8 -*-

import tensorflow as tf
from .Quantizer import Quantizer
from .QuantizerBase import DEFAULT_DATATYPE


class LinearQuantizer(Quantizer):
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
    def __init__(self, name, channel_wise_scaling=False, scale_inputs=False, dtype=DEFAULT_DATATYPE, internal_dtype=tf.dtypes.float32):
        super().__init__(name, channel_wise_scaling=channel_wise_scaling, scale_inputs=scale_inputs, dtype=dtype)
        self.internal_dtype = internal_dtype
        self.leak_clip = 0.0
    
    def setParams(self, min_value=None, max_value=None, q_steps=None, leak_clip=None):
        """ update/set quantisizer parameters
        
        important: after the parametes are set, call the model buildGraphs 
        method to apply the changes to the computation graph.
        
        Parameters:
            min_value (float):
                min value of the quantization range
                if None (default): parameter is not modified.
            max_value (float):
                max value of the quantization range.
                if None (default): parameter is not modified.
            q_steps (int):
                number of quantisation steps between the boundaries.
                if None (default): parameter is not modified.
            leak_clip (float):
                leak factor for backpropagation.
                when values are outside the quantsiation range (clipped to range)
                the gradient is multiplied by leak_clip
                if None (default): parameter is not modified.
        """
        super().setParams(min_value, max_value)
        if q_steps != None:
            self.q_steps = q_steps
        if leak_clip != None:
            self.leak_clip = leak_clip
            
        self.total_diff = self.max_value - self.min_value
        self.step_diff = self.total_diff / (self.q_steps - 1)

    def quant_forward(self, inputs):
        inputs = tf.cast(inputs, self.internal_dtype)
        inputs = inputs - self.zero_point
        inputs = inputs * self.scale

        x_clip = tf.clip_by_value(inputs, self.min_value, self.max_value)
        clipped_values_mask = tf.logical_or(x_clip == self.min_value, x_clip == self.max_value)
        tmp = (x_clip - self.min_value) / self.step_diff
        q_tmp = tf.math.round(tmp)
        y = q_tmp * self.step_diff + self.min_value

        y = y / self.scale
        y = y + self.zero_point
        y = tf.cast(y, self.dtype)
        return y, clipped_values_mask


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
            inputs_recast = tf.cast(inputs, self.internal_dtype)

            #tf.cond(self.is_scaled, lambda: 0.0, lambda: tf.py_function(self.findScale, inp=[inputs], Tout=tf.float32))
            y, _ = self.quant_forward(inputs_recast)


            # define the gradient calculation 
            def grad(dy):
                # test for every element of x if it is out of the bounds of 
                # the quantisation range
                is_out_of_range = tf.logical_or(inputs_recast < self.min_value, inputs_recast > self.max_value)
                # if is not out of range backpropagate dy
                # else backpropagate leak_clip * dy
                return tf.where(is_out_of_range, self.leak_clip * dy, dy)
            return y, grad

        return _quant(inputs)
