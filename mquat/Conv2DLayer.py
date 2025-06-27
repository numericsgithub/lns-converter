# -*- coding: utf-8 -*-

import tensorflow as tf

from .QEnums import QuantizerLocation
from .ComplexLayer import ComplexLayer
from .Utilities import createLoggerEntry
from .Variable import Variable
from .Conv2DOperationLayer import Conv2DOperationLayer
from .AddLayer import AddLayer
from .QuantizerBase import DEFAULT_DATATYPE

class Conv2DLayer(ComplexLayer):
    """2d convolution layer
    
    the 2d convolution layer applies the 2d convolution to the input,
    adds a bias for each filter and aplies a activation to the reslults.
    the 2d convolution and the bias addition are Operation layers that can be 
    modified.
    
    Parameters: 
        name (string):
            the name of the layer in the Tensorflow graph.
        filters (int):
            number of filters for the convolution.
        kernel_sizes (tuple of 2 ints):
            defines (height, width) of the kernel.
        strides (tuple of 2 ints):
            (y, x) offset of each convolution operation.
        padding (string):
            defines the padding algorithm:
            "SAME": 
                if the output should have the same size.
                if the mask moves over the border of the input tensor zeros 
                are assumed.
            "VALID": 
                if the filter mask is not alowed to move over the borders
                of the input tensor.
        kernel_initializer (TensorFlow init operation):
            operation that initialisizes the kernel variable.
        kernel_regularizer (TensorFlow regularization operation):
            operation that regularizes the kernel variable.
            None if deactivated.
        bias_initializer (TensorFlow init operation):
            operation that initialisizes the bias variable.
        bias_regularizer (TensorFlow regularization operation):
            operation that regularizes the bias variable
            None if deactivated.
        activation_func (function):
            actiavtion function of the layer.
            can be custom or from tf.nn.
        do_batch_norm (bool): 
            True if batch nomalization is needed.
        trainable_bias (bool):
            to not use the bias at all use None.
            Else true or false to have trainable or non-trainable bias
    
    Attributes:
        filters (int):
            number of filters for the convolution.
        kernel_sizes (tuple of 2 ints): 
            defines (height, width) of the kernel.
        strides (tuple of 2 ints):
            (y, x) offset of each convolution operation.
        f (Variable):
            the filter Variable of the convolution.
        b (Variable):
            the bias Variable
        conv2d (Conv2DOperationLayer):
            the convolution operation of the layer.
        bias_add (AddLayer):
            the bias addition operation of the layer.
    """
    def __init__(self, name, filters, kernel_sizes, strides=(1, 1), padding="VALID",
                 kernel_initializer="he_normal", kernel_regularizer=None,
                 bias_initializer="zeros", bias_regularizer=None,
                 activation_func=tf.nn.relu, do_batch_norm=True, trainable_bias=True, dtype=DEFAULT_DATATYPE):
        super().__init__(name, activation_func, do_batch_norm, dtype)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.f = Variable(name + "_f_var", kernel_initializer, kernel_regularizer, dtype=dtype)
        if trainable_bias is not None:
            self.b = Variable(name + "_b_var", bias_initializer, bias_regularizer, dtype=dtype, trainable=trainable_bias)
        else:
            self.b = None
        self.conv2d = Conv2DOperationLayer(name + "_conv_op", strides, padding, dtype=dtype)
        self.bias_add = AddLayer(name + "_bias_add_op", dtype=dtype)

    def getFilterCount(self):
        return self.filters

    def getStridesCount(self):
        return self.conv2d.strides[0]

    def create_logger_mapping(self):
        """
        Generates a dictionary holding the information about logging model
        Returns:

        """

        # result["weights"] = self.f.var
        # result["bias"] = self.b.var
        children = []
        children.append({"quant_in": self.quant_in.create_logger_mapping()})
        children.append({"weights": self.f.create_logger_mapping()})
        #children.append(self.conv2d.create_logger_mapping()) # todo implement all that later. Not that important
        if self.b != None:
            children.append({"bias": self.b.create_logger_mapping()})
            #children.append(self.bias_add.create_logger_mapping())
        # if self.batch_norm != None:
        #     children.append(self.batch_norm.create_logger_mapping())
        # if hasattr(self.activation, "create_logger_mapping"):
        #     children.append(self.activation.create_logger_mapping())
        # children.append(self.quant_out.create_logger_mapping())
        return createLoggerEntry(
            type_base_name="layer",
            type_name="conv",
            name=self.name,
            params={},
            children=children,
            loggers=[]
        )


    def getVariables(self):
        """get all variables of the layer
        
        Returns:
            (list of Variables):
                list contains filter and bias Variable.
        """
        variables = super().getVariables()
        if self.b != None:
            variables.extend([self.f, self.b])
        else:
            variables.extend([self.f])
        return variables


    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend(self.f.quant_out.getQuantVariables())
        if self.b is not None:
            variables.extend(self.b.quant_out.getQuantVariables())
        variables.extend(self.conv2d.mat_mul.quant_mul.getQuantVariables())
        variables.extend(self.conv2d.mat_mul.quant_out.getQuantVariables())
        variables.extend(self.conv2d.mat_mul.quant_sum.getQuantVariables())
        if self.b is not None:
            variables.extend(self.bias_add.quant_out.getQuantVariables())
        variables.extend(self.activation.getQuantVariables())
        variables.extend(self.quant_out.getQuantVariables())
        variables.extend(self.quant_in.getQuantVariables())
        return variables

    def getQuantizers(self, quant_location: QuantizerLocation = QuantizerLocation.EVERYWHERE):
        """get all quantizers of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        quantizers = []
        if quant_location == QuantizerLocation.AT_WEIGHTS or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.f.quant_out)

        if self.b is not None:
            if quant_location == QuantizerLocation.AT_BIASES or quant_location == QuantizerLocation.EVERYWHERE:
                quantizers.append(self.b.quant_out)

        if quant_location == QuantizerLocation.AT_MULTIPLICATION or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.conv2d.mat_mul.quant_mul)

        if quant_location == QuantizerLocation.AT_MULTIPLICATION or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.conv2d.mat_mul.quant_out)

        if quant_location == QuantizerLocation.AT_AFTER_SUM or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.conv2d.mat_mul.quant_sum)

        if quant_location == QuantizerLocation.AT_OUTPUT_ACTIVATIONS or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.bias_add.quant_out)

        if quant_location == QuantizerLocation.AT_OUTPUT_ACTIVATIONS or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.quant_out)

        if quant_location == QuantizerLocation.AT_INPUT_ACTIVATIONS or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.quant_in)
        return quantizers


    def build(self, input_shape):
        """build the variables of the layer
        
        called by TensorFlow before the first forward propagation.
        
        Parameters:
            input_shape (tuple of 4 ints):
                shape of the input tensor.
        """
        self.input_shape_ = input_shape
        input_channels = input_shape[3]
        self.f.build([*self.kernel_sizes, input_channels, self.filters])
        if self.b != None:
            self.b.build(self.filters)
        self.built = True

    def isQuantisizing(self):
        """is quantisizing
        
        test if the layer performs some quantisized operations.
        
        Returns:
            (bool): 
                true if an operation is quantisized.
        """
        is_quantisizing = super().isQuantisizing()
        is_quantisizing = is_quantisizing or self.conv2d.isQuantisizing()
        is_quantisizing = is_quantisizing or self.bias_add.isQuantisizing()
        return is_quantisizing

    def call(self, inputs, training):
        """forward propagation
        
        apply the layer operation in the forward pass.
        
        Parameters:
            inputs (list):
                list of all input tensors.
            training (bool):
                True: if in training mode.
                
        Returns:
            (tensor):
                the output of the layer.
        """
        # tf.print(self.name, "INPUT_SHAPE", tf.shape(inputs))
        tmp = self.quant_in(inputs)
        tmp = self.conv2d([tmp, self.f()])
        # test = tf.unique(tf.reshape(self.f(), [-1])).y
        # tf.print(self.name, tf.size(test), tf.sort(test), summarize=-1)
        if self.b != None:
            tmp = self.bias_add([tmp, self.b()])
        if self.batch_norm != None:
            tmp = self.batch_norm(tmp, training)
        tmp = self.activation(tmp)
        return self.quant_out(tmp)
    
    def saveStructureCustomFormat(self, folder_path, struct_file):
        """save the layer structure in custom format
    
        Parameters:
            folder_path (string):
                path to the folder to save the layer data.
            struct_file (file):
                file to save the structure data.
        """
        super().saveStructureCustomFormat(folder_path, struct_file)
        self.conv2d.saveStructureCustomFormat(folder_path, struct_file)
        self.bias_add.saveStructureCustomFormat(folder_path, struct_file)

    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.Conv2DLayer"
        config["filters"] = self.filters
        config["padding"] = self.conv2d.padding
        config["strides"] = self.conv2d.strides
        config["activation"] = self.activation.get_config()["activation"]
        config["input_shape"] = self.input_shape[1:4]
        config["batch_input_shape"] = self.input_shape
        config["kernel_size"] = self.kernel_sizes
        config["output_shape"] = self.output_shape # without batch size in contrast to dense layer where bath size exist
        config["data_format"] = "channels_last"
        config["quantize"] =  { "weight":    self.f.quant_out.get_config(), \
                                "bias":      self.b.quant_out.get_config(), \
                                "matmul_mul":self.conv2d.mat_mul.quant_out.get_config(), \
                                "matmul_sum":self.conv2d.mat_mul.quant_sum.get_config(), \
                                "matmul_out":self.conv2d.mat_mul.quant_out.get_config(), \
                                "bias_add":  self.bias_add.quant_out.get_config(),\
                                "out":       self.quant_out.get_config() \
                              }
        return config