# -*- coding: utf-8 -*-

import tensorflow as tf

from .OperationLayer import OperationLayer
from .QuantizerBase import DEFAULT_DATATYPE


class ConcatLayer(OperationLayer):
    """merge layer
    
    merges(conactenates) a list of input tensors along the second dimension
    to one tensor.
    
    Parameters: 
        name (string):
            the name of the layer in the TensorFlow graph.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations.
            default is float32.
    """
    def __init__(self, name, dtype=DEFAULT_DATATYPE):
        super().__init__(name, "concat", dtype=dtype)
        
    def call(self, inputs):
        """forward propagation
        
        apply the merge operation in the forward pass.
        
        Parameters:
            inputs (list):
                list of input tensors.
                
         Returns:
            (tensor):
                output of the layer.
        """
        tmp = tf.concat(inputs, 1)
        return self.quant_out(tmp)

        def get_config(self):
            config = super().get_config()
            config["type"] = "mquat.ConcatLayer"
            return config