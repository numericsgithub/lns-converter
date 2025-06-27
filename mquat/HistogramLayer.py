# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from .Layer import Layer
from .QuantizerBase import DEFAULT_DATATYPE


class HistogramLayer(Layer):
    def __init__(self, name, keep_axis_list=[], output_layer=None, dtype=DEFAULT_DATATYPE):
        super().__init__(name, False, dtype)
        self.output_layer = output_layer
        self.keep_axis_list = keep_axis_list
        self.x_min  = self.x_max  = self.x_n_bins  = None
        self.dy_min = self.dy_max = self.dy_n_bins = None
        self.active_statistics = True
        
    def isQuantisizing(self):
        return self.output_layer.isQuantisizing() if self.output_layer != None else False
    
    def activateStatistics(self):
        self.active_statistics = True
        
    def deactivateStatistics(self):
        self.active_statistics = False
    
    def setForwardParams(self, x_min, x_max, x_n_bins):
        self.x_min = x_min
        self.x_max = x_max
        self.x_n_bins = x_n_bins
        
    def setBackwardParams(self, dy_min, dy_max, dy_n_bins):
        self.dy_min = dy_min
        self.dy_max = dy_max
        self.dy_n_bins = dy_n_bins
    
    def resetStatistics(self):
        if hasattr(self, "x_histogram"):
            self.x_histogram.assign(0 * self.x_histogram)
        if hasattr(self, "dy_histogram"):
            self.dy_histogram.assign(0 * self.dy_histogram)
    
    def build(self, input_shape):
        self.keept_shape = []
        self.x_perms = [0]
        remaining_x_perms = []
        for axis, size in enumerate(input_shape[1:], 1):
            if axis in self.keep_axis_list:
                self.keept_shape.append(size)
                self.x_perms.append(axis)
            else:
                remaining_x_perms.append(axis)
        self.x_perms.extend(remaining_x_perms)
        
        if self.x_n_bins != None:
            self.x_histogram = tf.Variable(tf.zeros([*self.keept_shape, self.x_n_bins], tf.dtypes.int64), False, name="x_histogram")
        if self.dy_n_bins != None:
            self.dy_histogram = tf.Variable(tf.zeros([*self.keept_shape, self.dy_n_bins], tf.dtypes.int64), False, name= "dy_histogram")
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        if self.active_statistics and self.x_n_bins != None:
            x_transposed = tf.transpose(x, self.x_perms)
            x_flat = tf.reshape(x_transposed, [batch_size * np.prod(self.keept_shape), -1])
            histogram_x_array = tf.TensorArray(tf.dtypes.int64, size=0, dynamic_size=True)
            for i in tf.range(tf.shape(x_flat)[0]):
                histogram_x_i = tf.histogram_fixed_width(x_flat[i], [self.x_min, self.x_max], self.x_n_bins, tf.int64)
                histogram_x_array = histogram_x_array.write(i, histogram_x_i)  
            histogram_x_sum = tf.reduce_sum(histogram_x_array.stack(), 0)
            self.x_histogram.assign_add(tf.reshape(histogram_x_sum, self.x_histogram.shape))
            
        @tf.custom_gradient
        def _histogram_backward_fn(x):
            def _grad_fn(dy):
                if self.active_statistics and self.dy_n_bins != None:
                    dy_transposed = tf.transpose(dy, self.x_perms)
                    dy_flat = tf.reshape(dy_transposed, [batch_size * np.prod(self.keept_shape), -1])
                    histogram_dy_array = tf.TensorArray(tf.dtypes.int64, size=0, dynamic_size=True)
                    for i in tf.range(tf.shape(dy_flat)[0]):
                        histogram_dy_i = tf.histogram_fixed_width(dy_flat[i], [self.dy_min, self.dy_max], self.dy_n_bins, tf.int64)
                        histogram_dy_array = histogram_dy_array.write(i, histogram_dy_i)  
                    histogram_dy_sum = tf.reduce_sum(histogram_dy_array.stack(), 0)
                    self.dy_histogram.assign_add(tf.reshape(histogram_dy_sum, self.dy_histogram.shape))
                return dy
            return x, _grad_fn   
        
        return _histogram_backward_fn(self.output_layer(x))