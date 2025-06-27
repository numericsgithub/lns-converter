# -*- coding: utf-8 -*-

import tensorflow as tf

from .Reflectable import Reflectable
from .QuantizerBase import NON_QUANT
from .QuantizerBase import DEFAULT_DATATYPE


class Layer(tf.keras.layers.Layer, Reflectable):
    """base class of all layers
    
    the base layer class is derived from the keras base layer class.
    
    Parameters: 
           name (string):
               the name of the layer in the Tensorflow graph.
           trainable (bool):
               true if layer contains trainable variables.
    
    Attributes:
        quant_out (Quantizer):
            quantisizer at the output of the layer.
            default is NON_QUANT.
            specified after construction of the layer.
        trainable (Bool):
            true if the layer contains trainable parameters
    """
    def __init__(self, name, trainable=False, dtype=DEFAULT_DATATYPE):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self.quant_out = NON_QUANT
        self.quant_in = NON_QUANT
        self.trainable = trainable
        
        # prevent circular dependencies
        from .OperationLayer import OperationLayer
        from .ComplexLayer import ComplexLayer
        from .Variable import Variable
        self.OperationLayer = OperationLayer
        self.ComplexLayer = ComplexLayer
        self.Variable = Variable

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
        
    def getVariables(self):
        """get all variables of the layer
        
        has to be overwriten in the sublayers.
        
        Returns:
            (list of Varaiables):
                the base class returns an empty list.
        """
        return []

    def getVarialbesAsTensors(self):
        """get all variables of the layer as a tensor

        Returns:
            (list of Tensors):
                the base class returns an empty list.
        """
        mq_vars = self.getVariables()
        tensors = []
        for mqv in mq_vars:
            tensors.append(mqv.var)
        return tensors
    
    def isQuantisizing(self):
        """is quantisizing
        
        test if the layer performs some quantisized operations.
        
        Returns:
            (bool):
                True if output quantizier is active.
        """
        return self.quant_out.isQuantisizing()
    
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
        pass
    
    def getInputOperationNames(self):        
        # network input special case, when network is not build
        if len(self.inbound_nodes) == 0:
            return ["input"]
        
        input_layers = self.inbound_nodes[0].inbound_layers
        input_layers = input_layers if isinstance(input_layers, list) else [input_layers]
        input_names = []
        for input_layer in input_layers:
            if isinstance(input_layer, tf.keras.layers.InputLayer):
                input_names.append("input")
            elif isinstance(input_layer, self.OperationLayer):
                input_names.append(input_layer.name)
            elif isinstance(input_layer, self.ComplexLayer):
                input_names.append(input_layer.activation.name)
            elif isinstance(input_layer, self.Variable):
                input_names.append(input_layer.name)
        return input_names
    
    def saveStructureCustomFormat(self, folder_path, struct_file):
        """save the layer structure in custom format
        
        overwritten in the subclasses.
        
        Parameters:
            folder_path (string):
                path to the folder to save the layer data.
            struct_file (file):
                file to save the structure data.
        """
        pass

    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.Layer"
        return config