# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from .Layer import Layer
from .QuantizerBase import DEFAULT_DATATYPE


class OperationLayer(Layer):
    """operation layer
    
    base class for all operations in the network structure

    Parameters: 
        name (string):
            the name of the layer in the TensorFlow graph.
        op_name(string):
            the name of the operation type.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations and weights/variables.
            default is float32.
            
    Attributes:
        op_name(string):
               the name of the operation type.
    """
    def __init__(self, name, op_name, trainable=False, dtype=DEFAULT_DATATYPE):
        super().__init__(name, trainable, dtype)
        self.op_name = op_name        
            
    def call(self, inputs):
        """forward propagation
        
        overwritten in the subclasses to implement the functionality.
        
        Parameters:
            inputs (list):
                list of all input tensors.
        """
        pass
        
    def saveStructureCustomFormat(self, folder_path, struct_file):
        """save the layer structure in custom format
        
        overwritten in the subclasses.
        
        Parameters:
            folder_path (string):
                path to the folder to save the layer data.
            struct_file (file):
                file to save the structure data.
        """
        layer_file = self.writeCommonStructureCustomFormat(folder_path, struct_file)
        layer_file.close()
    
    def writeCommonStructureCustomFormat(self, folder_path, struct_file):
        """
        save the common informations for all subclasses

        Parameters:
            folder_path (string):
                path to the folder to save the layer data.
            struct_file (file):
                file to save the structure data.
                
        Returns:
            (file):
                layer file that contains all common layer informations.
                additional informations can be written in the subclasses
        """
        layer_file = open(os.path.join(folder_path, self.name), "w")
        layer_file.write("op_name: " + self.op_name + "\n")
        operations = self.getInputOperationNames()
        layer_file.write("num_inputs: " + str(len(operations)) + "\n")
        
        struct_file.write(self.name + ": " + " ".join(operations) + "\n")
        return layer_file

    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.OperationLayer"
        return config