# -*- coding: utf-8 -*-

import tensorflow as tf
from .OperationLayer import OperationLayer
from .MatMulOperationLayer import MatMulOperationLayer
from .QuantizerBase import DEFAULT_DATATYPE


class Conv2DOperationLayer(OperationLayer):
    """2d convolution operation layer
    
    the 2d convolution layer converts the convolution operation into a 
    matrix multiplication and reshapes the result back into an 4d tensor.
    
    Parameters: 
        name (string):
            the name of the layer in the Tensorflow graph.
        strides (tuple of 2 ints):
            (y, x) offset of each convolution operation.
        padding (string):
            defines the padding algorithm.
            "SAME": 
                if the output should have the same size.
                if the mask moves over the border of the input tensor zeros 
                are assumed.
            "VALID": 
                if the filter mask is not alowed to move over the borders
                of the input tensor.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations.
            default is float32.
    
    Attributes:
        strides (tuple of 2 ints):
            (y, x) offset of each convolution operation.
        padding (string):
            defines the padding algorithm.
            "SAME": 
                if the output should have the same size.
                if the mask moves over the border of the input tensor zeros 
                are assumed.
            "VALID": if the filter mask is not alowed to move over the borders
                of the input tensor.
        mat_mul (MatMulOperationLayer):
            the matrix multiplication operation layer.
    """
    def __init__(self, name, strides=(1, 1), padding='VALID', dtype=DEFAULT_DATATYPE):
        super().__init__(name, "conv2d", dtype=dtype)
        self.strides = strides
        self.padding = padding
        self.mat_mul = MatMulOperationLayer(name + "_matmul", dtype)
    
    def isQuantisizing(self):
        """is quantisizing
        
        test if the layer performs some quantisized operations.
        
        Returns:
            (bool):
                true if an operation is quantisized.
        """
        return self.mat_mul.isQuantisizing() or self.quant_out.isQuantisizing()
     

    def call(self, inputs):
        """forward propagation
        
        apply the layer operation in the forward pass.
        
        Parameters:
            inputs (list):
                list of all input tensors.
                
        Returns:
            (tensor):
                the output of the layer.
        """
        x, filters = inputs
        # performance optimization if matmul_quant is a NonQuantizer
        if not self.mat_mul.isQuantisizing():
            y_tmp = tf.nn.conv2d(x, filters, self.strides, self.padding)#, data_format="NCHW"
            return self.quant_out(y_tmp)
        # get input height and width
        [_, i_h, i_w, _] = x.shape
        # get filter height, width, num_in_channels, num_out_chanels
        [f_h, f_w, f_ci, f_co] = filters.shape
        # extract patches
        input_patches = tf.image.extract_patches(x, [1, f_h, f_w, 1], [1, *self.strides, 1], [1, 1, 1, 1], self.padding)
        [_, o_h, o_w, _] = input_patches.shape
        # reshape the image patches to a 2d-tensor (the first dimension (batch_size) is specified automatically)
        flat_patches = tf.reshape(input_patches, [-1, f_h * f_w * f_ci])
        # reshape the filter to 2d-tensor (the first dimension (batch_size) is specified automatically)
        flat_filters = tf.reshape(filters, [-1, f_co])
        y_tmp = self.mat_mul([flat_patches, flat_filters])
        # reshape the 2d result to 4D-Image Tensor (batch_size, height, width, num_channels)
        y_tmp = tf.reshape(y_tmp, [-1, o_h, o_w, f_co])
        return self.quant_out(y_tmp)
    
    def saveStructureCustomFormat(self, folder_path, struct_file):
        """save the layer structure in custom format
        
        Parameters:
            folder_path (string):
                path to the folder to save the layer data.
            struct_file (file):
                file to save the structure data.
        """
        layer_file = self.writeCommonStructureCustomFormat(folder_path, struct_file)
        layer_file.write("strides: " + " ".join(map(str, self.strides)) + "\n")
        layer_file.write("padding: " + self.padding.upper() + "\n")
        layer_file.close()

    def get_config(self):
        config = super().get_config()
        config["strides"] = self.strides
        config["padding"] = self.padding
        config["type"] = "mquat.Conv2DOperationLayer"
        return config