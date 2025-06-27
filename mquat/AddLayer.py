# -*- coding: utf-8 -*-

import tensorflow as tf

from .OperationLayer import OperationLayer
from .QuantizerBase import NON_QUANT, DEFAULT_DATATYPE


class AddLayer(OperationLayer):
    """addition layer
    
    performs an sequential accumulated addition on all inputs.

    Parameters: 
        name (string):
            the name of the layer in the TensorFlow graph.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations.
            default is float32. 
    Attributes:
        quant_add (Quantizer):
            quantisizer that quantisizes after each addition.
            default is NON_QUANT.
            specified after construction of the layer.
    """
    def __init__(self, name, dtype=DEFAULT_DATATYPE):
        super().__init__(name, "add", dtype=dtype)
        self.quant_add = NON_QUANT
        self.skipped_layer_name = None # placeholder
     
    def isQuantisizing(self):
        """is quantisizing
        
        test if the layer performs some quantisized operations.
        
        Returns:
            (bool): 
                True if add quantisizer or output quantizier is active.
        """
        return self.quant_add.isQuantisizing() or self.quant_out.isQuantisizing()

    def call(self, inputs):
        """forward propagation
        
        iterates over the list of input tensors and sums them up
        the result of each addition is quantisized.
        
        Parameters:
            inputs(list): 
                list of input tensors.
                
        Returns:
            (TensorFlow tensor): 
                output of the layer.
        """
        tmp = inputs[0]
        for input in inputs[1:]:
            #tmp = tf.transpose(tmp, perm=(0,2,3,1))
            tmp = self.quant_add(tmp + input)
            #tmp = tf.transpose(tmp, perm=(0,3,1,2))
        return self.quant_out(tmp)