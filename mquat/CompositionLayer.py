# -*- coding: utf-8 -*-

import tensorflow as tf
from .QuantizerBase import DEFAULT_DATATYPE

from .Layer import Layer


class CompositionLayer(Layer):
    """composition layer
    TODO DO DOCUMENTATION HERE!
    the complex layer contains an activation function and optionally a
    batch normalization layer
    the functionality is created in the subclasses
    
    Parameters: 
        name (string):
            the name of the layer in the TensorFlow graph.
        activation_func (function):
            actiavtion function of the layer.
            can be custom or from tf.nn.
        do_batch_norm (bool):
            true if batch nomalization is needed.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations and weights/variables.
            default is float32.
    
    Attributes:
        activation (ActivationLayer):
            activation layer
        batch_norm (BatchNormalization):
            applied before or after the activation function.
            can be None if not needed.
    """
    def __init__(self, name, trainable=False, dtype=DEFAULT_DATATYPE):
        super().__init__(name, trainable, dtype)

    def getVariables(self):
        """get all variables of the layer.
        
        Returns:
            (list of Varaiables):
        """
        return []

    def getSubLayers(self):
        """get all sub-layers of the layer.

        Returns:
            (list of Layers):
        """
        return []
    
    def isQuantisizing(self):
        """is quantisizing
        
        test if the layer performs some quantisized operations.
        
        Returns:
            (bool): 
                true if an operation is quantisized.
        """
        pass
    
    def saveStructureCustomFormat(self, folder_path, struct_file):
        """save the layer structure in custom format
    
        for complex layers it has to be overwritten in the subclasses to save 
        the containing layers.
    
        Parameters:
            folder_path (string):
                path to the folder to save the layer data.
            struct_file (file):
                file to save the structure data.
        """
        pass