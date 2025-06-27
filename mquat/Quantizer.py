# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

# from .LoggingLayer import LoggingLayer
from .QuantizerBase import QuantizerBase
from .QuantizerBase import DEFAULT_DATATYPE
import matplotlib.pyplot as plt

from .Utilities import createLoggerEntry


class Quantizer(QuantizerBase):
    """base class for all quantisizers except the NON_QUANT

    Parameters:
        name (string):
            the name of the layer in the TensorFlow graph.
        dtype (tf.dtypes.DType):
            datatype of the layers's operations.
            default is float32.

    Attributes:
        active(bool):
            default value is True (enabled).
    """

    def __init__(self, name, channel_wise_scaling=False, scale_inputs=False, dtype=DEFAULT_DATATYPE, pre_filters=[], post_filters=[], hirachy_value=None):
        super().__init__(name, dtype, hirachy_value=hirachy_value)
        # set the inital value to true (enabled)
        self.active = True
        self.scale = None
        self.scale_error = None
        self.zero_point = None
        self.is_scaled = tf.Variable(not scale_inputs, trainable=False, name=name + "_is_scaled")
        self.channel_wise_scaling = channel_wise_scaling
        #self.log_before_quantization = LoggingLayer(name + "_log_before_quantization", name + "_log_before_quantization.gz")
        # self.log_after_quantization = LoggingLayer(name + "_log_after_quantization", name + "_log_after_quantization.gz")
        self.pre_filters = []
        self.pre_filters.extend(pre_filters)
        self.post_filters = []
        self.post_filters.extend(post_filters)
        self.active_var = None

    def create_logger_mapping(self):
        return createLoggerEntry(
            type_base_name="quantizer",
            type_name="quantizer",
            name=self.name,
            params={},
            children=[],
            loggers=[] # {"After quantization": self.log_after_quantization.name}
        )

    def build(self, input_shape):
        if self.channel_wise_scaling:
            scale_init = tf.ones(input_shape[-1])
            zero_point_init = tf.zeros(input_shape[-1])
        else:
            scale_init = 1.0
            zero_point_init = 0.0
        self.scale = tf.Variable(scale_init, trainable=False, name=self.name + "_scale")
        self.scale_error = tf.Variable(scale_init, trainable=False, name=self.name + "_scale_error")
        self.zero_point = tf.Variable(zero_point_init, trainable=False, name=self.name + "_zero_point")
        self.active_var = tf.Variable(self.active, trainable=False, name=self.name + "_active_var")

    def getQuantVariables(self):
        return [self.scale, self.scale_error, self.zero_point, self.is_scaled]

    def setParams(self, min_value=None, max_value=None):
        """ update/set quantisizer parameters

        important: after the parametes are set, call the model buildGraphs
        method to apply the changes to the computation graph.

        Parameters:
            min_value (float):
                min value of the quantization range.
                if None (default): parameter is not modified.
            max_value (float):
                max value of the quantization range.
                if None (default): parameter is not modified.
        """
        if min_value != None:
            self.min_value = min_value
        if max_value != None:
            self.max_value = max_value

    def activate(self):
        """enable the quantisizer
        """
        self.active = True
        if self.active_var is not None:
            self.active_var.assign(True)

    def deactivate(self):
        """disable the quantisizer
        """
        self.active = False
        if self.active_var is not None:
            self.active_var.assign(False)

    def quant_forward(self, inputs):
        return inputs

    def isQuantisizing(self):
        """test if the quantisizer is quantisizing

        overwriten in subclasses.

        Returns:
            (bool):
                True if quantisizer is active.
        """
        return self.active
    #
    # def findScale(self, inputs, scale_overwrite=None):
    #     if self.is_scaled.numpy() == False:
    #
    #         self.is_scaled.assign(True)
    #         inputs = tf.cast(inputs, dtype=tf.float32)
    #
    #         if self.channel_wise_scaling:
    #             channel_dim = len(tf.shape(inputs)) - 1
    #             other_dims = [x for x in range(channel_dim)]
    #
    #             zero_point = tf.math.reduce_mean(inputs, axis=other_dims)
    #             # zero_point = tf.broadcast_to(zero_point, tf.shape(inputs))
    #             inputs = inputs - zero_point
    #
    #             max_per_channel = tf.reduce_max(tf.abs(inputs), axis=other_dims)
    #             tf.print("SHAPE INPUTS", tf.shape(inputs))
    #             tf.print("SHAPE tf.reduce_max(tf.abs(inputs), axis=other_dims)", tf.shape(max_per_channel))
    #             scales = tf.ones_like(max_per_channel) * 1.0
    #             scales = scales / max_per_channel
    #
    #             scales = tf.where(tf.math.is_nan(scales), 1e-8, scales)
    #             scales = tf.where(tf.math.is_inf(scales), 1e-8, scales)
    #             scales = tf.clip_by_value(scales, 1e-8, 1e+8)
    #             # scales = tf.broadcast_to(scales, tf.shape(inputs))
    #             tf.print("scales", scales)
    #             self.scale.assign(scales)
    #             self.zero_point.assign(zero_point)
    #         else:
    #             zero_point = tf.math.reduce_mean(inputs)
    #             inputs = inputs - zero_point
    #
    #             max_per_channel = tf.reduce_max(tf.abs(inputs))
    #             scales = 1.0 / max_per_channel
    #
    #             scales = tf.where(tf.math.is_nan(scales), 1e-8, scales)
    #             scales = tf.where(tf.math.is_inf(scales), 1e-8, scales)
    #             scales = tf.clip_by_value(scales, 1e-8, 1e+8)
    #             self.scale.assign(scales)
    #             self.zero_point.assign(zero_point)
    #         tf.print("INFO EVERYTHING FOR inputs", self.name, inputs, tf.reduce_min(inputs), tf.reduce_max(inputs))
    #         tf.print("INFO EVERYTHING FOR inputs", self.name, inputs, tf.reduce_min(inputs), tf.reduce_max(inputs))
    #         tf.print("SCALED EVERYTHING FOR", self.name, self.scale, self.zero_point)
    #     return 0.0

    def call(self, inputs, quantisize=None):
        """forward propagation

        if the quantisizer is active it calls the quant function on the input
        and returns the result else it returns the input.

        Parameters:
            inputs (list of tensors):
                list of all input tensors.
            quantisize (None or True or False):
                None (Default):
                    use internal active state of the quantisizer.
                True:
                    force quantization of the input (ignore the actual active state).
                False:
                    force direct return of the input (ignore the actual active state).
        Returns:
            (tensor):
                quantisized imput.
        """
        if quantisize == True or self.isQuantisizing():
            result = self.quant(inputs)
            # self.log_after_quantization(result)
            return result
        return inputs

    def quant(self, inputs):
        """quantisation function

        overwriten in the actual quantisizer.

        Parameters:
            inputs (list):
                list of all input tensors.

        Returns:
            (tensor): 
                quantisized input.
        """
        return inputs

    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.Quantizer"
        config["quant_data"] = {"active": self.active}
        return config