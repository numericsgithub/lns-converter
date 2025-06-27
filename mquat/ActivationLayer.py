# -*- coding: utf-8 -*-

import tensorflow as tf

from .OperationLayer import OperationLayer
from .QuantizerBase import DEFAULT_DATATYPE


class ActivationLayer(OperationLayer):
    """activation layer
    
    Parameters: 
        name (string):
            the name of the layer in the TensorFlow graph.
        activation (tf.nn)
            activation function from tf.nn or custom.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations.
            default is float32.
    Attributes:
        activation (tf.keras.layers.Activation)
    """
    def __init__(self, name, activation, dtype=DEFAULT_DATATYPE):
        super().__init__(name, "activation", dtype=dtype)
        self.activation_desc = activation
        self.activation = tf.keras.layers.Activation(activation, dtype=dtype)

    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend(self.quant_out.getQuantVariables())
        variables.extend(self.quant_in.getQuantVariables())
        return variables

    def call(self, inputs):
        """forward propagation
        
        apply the activation function in the forward pass.
        
        Parameters:
            inputs (list):
                list of input tensors.
                
        Returns:
            (tensor): 
                output of the layer.
        """
        tmp = self.quant_in(inputs)
        tmp = self.activation(tmp)
        return self.quant_out(tmp)
    
    def saveStructureCustomFormat(self, folder_path, struct_file):
        """save the layer structure in custom format
        
        Parameters:
            folder_path (string):
                path to the folder to save the structure data.
            struct_file (file):
                file to save the structure data.
        """        
        layer_file = self.writeCommonStructureCustomFormat(folder_path, struct_file)
        layer_file.write("activation_func:" + self.activation.activation.__name__.upper())
        layer_file.close()

    def get_config(self):
        config =super().get_config()
        config["activation"] = self.activation.activation.__name__.upper()
        config["type"] = "mquat.ActivationLayer"
        return config