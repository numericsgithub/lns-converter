# -*- coding: utf-8 -*-

import tensorflow as tf
from .ComplexLayer import ComplexLayer
from .Conv2DLayer import Conv2DLayer
from .AddLayer import AddLayer
import tensorflow.keras.layers as layers
from .ActivationLayer import ActivationLayer


class ResidualBlockLayer50(ComplexLayer):
    """resiudal  block layer
    
    special layer that is used in ResNets.
    
    Parameters: 
        name (string): 
            the name of the model in the TensorFlow graph.
        channels_list (tuple of 2 ints):
            number of output channels. The first int describes the channels for the first
            two conv layers. The second one the last conv layer.
        strides (tuple of 2 ints):
            (y, x) offset of each convolution operation.
        kernel_initializer (TensorFlow init operation):
            operation that initialisizes the kernel variables.
        kernel_regularizer (TensorFlow regularization operation):
            operation that regularizes the kernel variables.
            None if deactivated.
        bias_initializer (TensorFlow init operation):
            operation that initialisizes the bias variables.
        bias_regularizer (TensorFlow regularization operation):
            operation that regularizes the bias variables
            None if deactivated.
        activation_func (function):
            actiavtion function of the layer.
            can be custom or from tf.nn.
        do_batch_norm (bool): 
            true if batch nomalization is needed.
    """
    def __init__(self, name, channels_list, first_stride, has_conv_shortcut=False,
                 kernel_initializer="he_normal", kernel_regularizer=None,
                 bias_initializer="zeros", bias_regularizer=None,
                 activation_func=tf.nn.relu, do_batch_norm=True, tainable_bias=False):
        super().__init__(name, activation_func, False)
        self.conv_1 = Conv2DLayer(name + "_conv_1", channels_list[0], (1, 1), strides=first_stride,
                                  activation_func=activation_func, do_batch_norm=do_batch_norm, tainable_bias=tainable_bias)
        self.conv_2 = Conv2DLayer(name + "_conv_2", channels_list[0], (3, 3), strides=(1, 1), padding="VALID",
                                  activation_func=activation_func, do_batch_norm=do_batch_norm, tainable_bias=tainable_bias)
        self.conv_3 = Conv2DLayer(name + "_conv_3", channels_list[1], (1, 1), strides=(1, 1),
                                  activation_func=tf.identity, do_batch_norm=do_batch_norm, tainable_bias=tainable_bias)
        # if the first_stride is greater than 1 the convolution changes the size of the
        # tensors, to add the shortcut (input tensor) it also must be resized by a convoloution
        if has_conv_shortcut:
            self.conv_short = Conv2DLayer(name + "_conv_short", channels_list[1], (1, 1), strides=first_stride, padding="VALID",
                                          activation_func=tf.identity, do_batch_norm=do_batch_norm, tainable_bias=tainable_bias)
        else:
            # if conv_short is None no convolution is applied to shortcut(input tensor)
            self.conv_short = None
        self.input_add = AddLayer(name + "_input_add_op")
    
    def getVariables(self):
        """get all variables of the layer
        
        Returns:
            (list of Variables):
                list contains filter and bias Variable.
        """
        variables = []
        variables.extend(self.conv_1.getVariables())
        variables.extend(self.conv_2.getVariables())
        variables.extend(self.conv_3.getVariables())
        if self.conv_short != None:
            variables.extend(self.conv_short.getVariables())
        return variables

    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend(self.conv_1.getQuantVariables())
        variables.extend(self.conv_2.getQuantVariables())
        variables.extend(self.conv_3.getQuantVariables())
        if self.conv_short != None:
            variables.extend(self.conv_short.getQuantVariables())
        return variables


    def getSubLayers(self):
        layers = []
        layers.append(self.conv_1)
        layers.append(self.conv_2)
        layers.append(self.conv_3)
        if self.conv_short != None:
            layers.append(self.conv_short)
        return layers

    def isQuantisizing(self):
        """test if the layer performs some quantisized operations
        
        Returns:
            (bool): 
                true if an operation is quantisized.
        """
        return self.conv_1.isQuantisizing() or self.conv_2.isQuantisizing() or self.conv_3.isQuantisizing() or self.input_add.isQuantisizing() or self.quant_out.isQuantisizing()

    def call(self, inputs):
        """forward propagation
        
        apply the layer operations in the forward pass.
        
        Parameters:
            inputs (list): 
                list of all input tensors.
                
        Returns:
            (tensor): 
                the output of the layer.
        """
        tmp = self.conv_1(inputs)
        tmp = tf.pad(tmp, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        tmp = self.conv_2(tmp)
        tmp = self.conv_3(tmp)
        if self.conv_short != None:
            inputs = self.conv_short(inputs)
        tmp = self.input_add([tmp, inputs])
        tmp = tf.nn.relu(tmp)
        return self.quant_out(tmp)
    
    def saveStructureCustomFormat(self, folder_path, struct_file):
        """save the layer structure in custom format
    
        Parameters:
            folder_path (string):
                path to the folder to save the layer data.
            struct_file (file):
                file to save the structure data.
        """
        self.conv_1.saveStructureCustomFormat(folder_path, struct_file)
        self.conv_2.saveStructureCustomFormat(folder_path, struct_file)
        self.conv_3.saveStructureCustomFormat(folder_path, struct_file)
        if self.conv_short != None:
            self.conv_short.saveStructureCustomFormat(folder_path, struct_file)
        self.input_add.saveStructureCustomFormat(folder_path, struct_file)
        self.activation.saveStructureCustomFormat(folder_path, struct_file)

    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.ResidualBlockLayer50"
        return config