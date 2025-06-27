# -*- coding: utf-8 -*-

import tensorflow as tf

from .OperationLayer import OperationLayer
from .Variable import Variable
import warnings
from .QuantizerBase import DEFAULT_DATATYPE


class BatchNormalization(OperationLayer):
    """batch normalization layer

    uses the keras.layers.BatchNormalization layer and wraps variables around it.
    
    Parameters: 
        name (string):
            the name of the layer in the TensorFlow graph.
            
    Attributes:
        batch_norm (tf.keras.layers.BatchNormalization):
            keras batch normalization layer.
    """

    def __init__(self, name, variance_epsilon=0.0000001, dtype=DEFAULT_DATATYPE):
        super().__init__(name, "batch_norm", True, dtype=dtype)
        # the dynamic range of floats with less than 32 bits is to low so
        # perform internal operations at least with 32 bit variables
        batch_norm_dtype = dtype
        # self.keras_batch_norm = tf.keras.layers.BatchNormalization(axis, momentum, epsilon,
        #                                                            name=f"{name}_keras_batch_norm", dtype=batch_norm_dtype)
        self.beta = Variable(f"{name}_beta_var", "zeros", trainable=True, dtype=batch_norm_dtype)
        self.gamma = Variable(f"{name}_gamma_var", "ones", trainable=True, dtype=batch_norm_dtype)
        self.moving_mean = Variable(f"{name}_moving_mean_var", "zeros", trainable=False, dtype=batch_norm_dtype)
        self.moving_variance = Variable(f"{name}_moving_variance_var", "ones", trainable=False, dtype=batch_norm_dtype)
        self.variance_epsilon = variance_epsilon

    def getVariables(self):
        """get all variables of the layer.
        
        Returns:
            (list of Varaiables):
        """
        return [self.beta, self.gamma, self.moving_mean, self.moving_variance]
               
    def build(self, input_shape):
        """build the variables of the layer.
        
        Parameters:
            input_shape (tuple of 2 ints):
                shape of the input tensor.
        """
        # use the Keras Batch Norm Layers Variables internally in the mquat Variables
        # to apply quantizations to them.
        # set the mquat Variables as built to prevent overwrites.  
        # self.keras_batch_norm.build(input_shape)
        self.beta.build([input_shape[-1]])
        self.gamma.build([input_shape[-1]])
        self.moving_mean.build([input_shape[-1]])
        self.moving_variance.build([input_shape[-1]])

        # self.keras_batch_norm.beta = self.beta.var
        # self.keras_batch_norm.gamma = self.gamma.var
        # self.keras_batch_norm.moving_mean = self.moving_mean.var
        # self.keras_batch_norm.moving_variance = self.moving_variance.var

        # self.beta.var = self.keras_batch_norm.beta
        # self.beta.built = True
        # self.gamma.var = self.keras_batch_norm.gamma
        # self.gamma.built = True
        # self.moving_mean.var = self.keras_batch_norm.moving_mean
        # self.moving_mean.built = True
        # self.moving_variance.var = self.keras_batch_norm.moving_variance
        # self.moving_variance.built = True
      
    def assign_moving_mean(self, value, **kwargs):
        self.moving_mean.var.assign(value, **kwargs)

    def assign_moving_variance(self, value, **kwargs):
        self.moving_variance.var.assign(value, **kwargs)   
    
    def call(self, inputs, training):
        """forward propagation
        
        apply the layer operation in the forward pass.
        
        Parameters:
            inputs (list):
                list of all input tensors.
            training (bool):
                true: if in training mode.
                
        Returns:
            (tensor):
                the output of the layer.
        """
        return tf.nn.batch_normalization(inputs, self.moving_mean(), self.moving_variance(), self.beta(), self.gamma(), variance_epsilon=self.variance_epsilon)
        # print("BNORM 1", inputs.dtype)
        # # keep a reference to the keras batch norm layer variables
        # tmp_beta = self.keras_batch_norm.beta
        # tmp_gamma = self.keras_batch_norm.gamma
        # tmp_moving_mean = self.keras_batch_norm.moving_mean
        # tmp_moving_variance = self.keras_batch_norm.moving_variance
        #
        # print("BNORM 2", inputs.dtype)
        # # temporary overwrite the keras batch norm layer variables with the quantized variables
        # self.keras_batch_norm.beta = self.beta()
        # self.keras_batch_norm.gamma = self.gamma()
        # self.keras_batch_norm.moving_mean = self.moving_mean()
        # self.keras_batch_norm.moving_variance = self.moving_variance()
        # print("BNORM 5", inputs.dtype)
        # # add a assign method to the mean and variance Tensors to allow updates to the internal state
        # if not hasattr(self.keras_batch_norm.moving_mean, "assign"):
        #     self.keras_batch_norm.moving_mean.assign = self.assign_moving_mean
        # if not hasattr(self.keras_batch_norm.moving_variance, "assign"):
        #     self.keras_batch_norm.moving_variance.assign = self.assign_moving_variance
        # # call the internal keras batch norm layer
        # outputs = self.keras_batch_norm(inputs, training=training)
        #
        # print("BNORM 2", outputs.dtype)
        # # undo the overwrite
        # self.keras_batch_norm.beta = tmp_beta
        # self.keras_batch_norm.gamma = tmp_gamma
        # self.keras_batch_norm.moving_mean = tmp_moving_mean
        # self.keras_batch_norm.moving_variance = tmp_moving_variance
        #
        # return outputs

        def get_config(self):
            config = super().get_config()
            config["type"] = "mquat.BatchNormalization"
            return config