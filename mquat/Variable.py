# -*- coding: utf-8 -*-

import tensorflow as tf

from .QuantizerBase import NON_QUANT
from .Layer import Layer
import os
import numpy as np
import mquat.SQUANT.squant as sq
from .QuantizerBase import DEFAULT_DATATYPE
from .Utilities import createLoggerEntry


class Variable(Layer):
    """variable
    
    the Variable contains a network parameter.
    it has methods to load and save.
    at the output it has a quantisizer that can contrain the parameter range
    of the variable
    
    Parameters: 
        name (string):
            the name of the variable in the TensorFlow graph.
        initializer (TensorFlow init operation):
            operation that initialisizes the variable.
        regularizer (TensorFlow regularization operation):
            None if deactivated.
        trainable (bool):
            True if the variable should be trained by the optimizer.
        dtype (tf.dtypes.DType):
            datatype of the layer's weights/variable.
            default is float32.

    Attributes:
        initializer (TensorFlow init operation):
            operation that initialisizes the variable.
        regularizer (TensorFlow regularization operation):
            None if deactivated.
    """
    
    def __init__(self, name, initializer=None, regularizer=None, 
                 trainable=True, dtype=DEFAULT_DATATYPE):
        super().__init__(name, trainable, dtype)
        self.initializer = initializer
        self.regularizer = regularizer

    def create_logger_mapping(self):
        return createLoggerEntry(
            type_base_name="variable",
            type_name="variable",
            name=self.name,
            params={},
            children=[{"quant_out": self.quant_out.create_logger_mapping()}],
            loggers=[]
        )

    def build(self, shape):
        """build the TensorFlow graph
                
        Parameters: 
            shape (tuple of int):
                shape of the  variable.
        """
        self.var = self.add_weight(self.name, shape=shape, initializer=self.initializer,
                                   regularizer=self.regularizer, trainable=self.trainable,
                                   constraint=self.quant_out.createConstraint(self.dtype),
                                   dtype=self.dtype)
        self.built = True

    def call(self, inputs=None):
        """forward propagation
                
        Parameters: 
            inputs (tensor):
                allways None, only needed to match the method of the parent class.
        
        Returns:
            (tensor):
                quantisized output of the variable.
        """
        return self.quant_out(self.var)
        # if tf.math.equal(self.var, self.var_hash):
        #     return self.var_quant_cache + 0.0
        # else:
        #     self.var_hash.assign(self.var)
        #     self.var_quant_cache.assign(self.quant_out(self.var))
        # return self.var_quant_cache + 0.0

    def __call__(self):
        """call as function (forward propagation)
                
         Returns:
            (tensor):
                quantisized output of the variable.
        """
        return super().__call__([])
    
    def getValuesNP(self, quantisize=False):
        """get variable values as numpy array

        Parameters:
            quantisize (None or True or False):
                 None (Default):
                     use internal active state of the quantisizer (quant_out) in the variable.
                 True:
                     force quantization of the variable.
                 False:
                     force no quantization of the variable.
        Returns:
            (np.array):
                values of the variable in a numpy array.
        """
        if quantisize:
            return self.quant_out(self.var, quantisize=quantisize).numpy()
        else:
            return self.var.numpy()

    def saveCustomFormat(self, folder_path, gz_compress=False, quantisize=False):
        """save variable to numpy savetxt format
                
        Parameters: 
            folder_path (string):
                path to the folder where to save the variable in numpy 
                savetxt format.
            struct_file (text file):
                file with the structure of the network (connections between layers).
            gz_compress (bool):
                if true gz compression will be used.
            quantisize (bool):
                if true the variable will be saved quantisized.
        """
        file_extension = ".gz" if gz_compress else ".csv"
        values_np = self.getValuesNP(quantisize)
        header = "shape: " + " ".join(map(str, values_np.shape))
        file_path = os.path.join(folder_path, self.name + file_extension)
        np.savetxt(file_path, values_np.flatten(), delimiter='\n', header=header)
    
    def loadCustomFormat(self, folder_path, gz_compressed=False):
        """load variable from numpy savetxt format
                
        Parameters: 
            folder_path (string):
                path to the folder where the variable was saved.
            gz_compressed (bool):
                must be true if the variable was saved compressed.
        """
        file_extension = ".gz" if gz_compressed else ".csv"
        file_path = os.path.join(folder_path, self.name + file_extension)
        values_np = np.loadtxt(file_path, dtype=self.var.numpy().dtype, delimiter='\n')
        self.loadNumpyArray(values_np)

    def loadNumpyArray(self, numpy_array):
        self.var.assign(numpy_array.reshape(self.var.shape))

    def applySQUANT(self):
        """
        Automatically applies squant to the variable.
        WARNING: The original full-precision weights will be overwritten!
        Returns:

        """


        dtype = self.var.dtype

        x = tf.cast(self.var, dtype=DEFAULT_DATATYPE)  # The unrounded weights
        if tf.size(tf.shape(x)) != 4 and tf.size(tf.shape(x)) != 2:
            return
        if self.quant_out is NON_QUANT:
            return
        scale = 1.0
        scale *= tf.cast(self.quant_out.scale, dtype=DEFAULT_DATATYPE).numpy()

        zero_point = 0.0
        zero_point += tf.cast(self.quant_out.zero_point, dtype=DEFAULT_DATATYPE).numpy()
        self.quant_out.scale.assign(tf.ones_like(self.quant_out.scale))
        self.quant_out.zero_point.assign(tf.zeros_like(self.quant_out.zero_point))

        x = x - zero_point
        x = x * scale

        xquant = tf.cast(self.quant_out.quant(x), DEFAULT_DATATYPE)  # The rounded weights "round to nearest"
        coeffs = tf.cast(self.quant_out.getCoeffs(), DEFAULT_DATATYPE) # All possible coefficients allowed by the quantization


        #scale = 1.0 # self.quant_out.getCoeffsScaling()
        is_non_uniform = self.quant_out.isNonUniform()

        #x = tf.clip_by_value(x, tf.reduce_min(x), tf.reduce_max(x))

        # x *= scale
        # xquant *= scale
        # coeffs *= scale


        # for coeff in coeffs:
        #     print("COEFF2: {:.20f}, ".format(coeff))

        # print("scaled coeffs are", coeffs)

        # transpose to torch format
        if tf.size(tf.shape(x)) == 4:
            x = tf.transpose(x, [3, 2, 0, 1])
            xquant = tf.transpose(xquant, [3, 2, 0, 1])
        else:
            x = tf.transpose(x, [1, 0])
            xquant = tf.transpose(xquant, [1, 0])

        xquant = xquant.numpy()
        x = x.numpy()
        coeffs = coeffs.numpy()

        xsquant, results_abs_sum, results_sum = sq.adaptive_round(self.name, tf.cast(x, tf.float32).numpy(), tf.cast(xquant, tf.float32).numpy(), tf.cast(coeffs, tf.float32).numpy(), isNonUniform=is_non_uniform) # The rounded weights using squant
        xsquant = tf.cast(xsquant, DEFAULT_DATATYPE)
        results_abs_sum = tf.cast(results_abs_sum, DEFAULT_DATATYPE)
        results_sum = tf.cast(results_sum, DEFAULT_DATATYPE)

        # transpose to tensorflow format
        if tf.size(tf.shape(x)) == 4:
            xsquant = tf.transpose(xsquant, [2, 3, 1, 0])
        else:
            xsquant = tf.transpose(xsquant, [1, 0])

        xsquant = (xsquant / scale) + zero_point
        # todo Hey maybe this is not so great actually. The float weights are overwritten ... Maybe implement a squant quantization layer later on...
        self.var.assign(tf.cast(xsquant, dtype)) # apply the squant weights by brutally overwriting those nifty floaty pointy high resolutionary weightis.
        self.quant_out.scale.assign(scale)
        self.quant_out.zero_point.assign(zero_point)

