# -*- coding: utf-8 -*-

import tensorflow as tf

from .CompositionLayer import CompositionLayer
from .ActivationLayer import ActivationLayer
from .BatchNormalization import BatchNormalization
from .QuantizerBase import DEFAULT_DATATYPE


class ComplexLayer(CompositionLayer):
    """complex layer
    
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
    def __init__(self, name, activation_fn, do_batch_norm, dtype=DEFAULT_DATATYPE):
        super().__init__(name, True, dtype)
        self.activation = ActivationLayer(name + "_activation_op", activation_fn, dtype)
        if do_batch_norm == True:
            self.batch_norm:BatchNormalization = BatchNormalization(name + "_batch_norm_op", dtype=dtype) #  epsilon=1e-5,
        else:
            self.batch_norm:BatchNormalization = None

    def getFilterCount(self):
        pass

    def getStridesCount(self):
        pass

    def getVariables(self):
        """get all variables of the layer.
        
        Returns:
            (list of Varaiables):
        """
        variables = self.batch_norm.getVariables() if self.batch_norm != None else [] 
        return variables
    
    def isQuantisizing(self):
        """is quantisizing
        
        test if the layer performs some quantisized operations.
        
        Returns:
            (bool): 
                true if an operation is quantisized.
        """
        is_quantisizing = self.quant_out.isQuantisizing()
        if self.batch_norm != None:
            is_quantisizing = is_quantisizing or self.batch_norm.isQuantisizing()
        return is_quantisizing
    
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
        if self.batch_norm != None:
            self.batch_norm.saveStructureCustomFormat(folder_path, struct_file)
        self.activation.saveStructureCustomFormat(folder_path, struct_file)

    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.ComplexLayer"
        return config