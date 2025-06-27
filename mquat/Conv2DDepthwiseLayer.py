# -*- coding: utf-8 -*-

import tensorflow as tf

from .Conv2DOperationLayer import Conv2DOperationLayer
from .ComplexLayer import ComplexLayer
from .Variable import Variable
from .AddLayer import AddLayer
from .QuantizerBase import DEFAULT_DATATYPE


class Conv2DDepthwiseLayer(ComplexLayer):
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
        activation (function):
            actiavtion function of the layer
            can be custom or from tf.nn.
        do_batch_norm (bool): 
            True if batch nomalization is needed.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations and weights/variables.
            default is float32.
    
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
    def __init__(self, name, kernel_sizes, strides=(1, 1), padding="VALID",
                 kernel_initializer="he_normal", kernel_regularizer=None,
                 bias_initializer="zeros", bias_regularizer=None, trainable_bias=True,
                 activation=tf.nn.relu, do_batch_norm=True, dtype=DEFAULT_DATATYPE, in_channels=None):
        super().__init__(name, activation, do_batch_norm, dtype)
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        # self.f = []
        if trainable_bias is not None:
            self.b = Variable(name + "_b_var", bias_initializer, bias_regularizer, dtype=dtype, trainable=trainable_bias)
        else:
            self.b = None
        self.conv2d = Conv2DOperationLayer(name + "_conv_op", strides, padding, )
        self.bias_add = AddLayer(name + "_bias_add_op")
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.f = Variable(self.name + "_f", self.kernel_initializer, self.kernel_regularizer, dtype=dtype, trainable=True)
        # for i in range(in_channels):
        #     self.f.append(Variable(self.name + "_f_"+str(i), self.kernel_initializer, self.kernel_regularizer))

    def getFilterCount(self):
        return self.in_channels

    def getStridesCount(self):
        return self.conv2d.strides[0]

    def getVariables(self):
        """get all variables of the layer
        
        Returns:
            (list of Variables):
                list contains filter and bias Variable.
        """
        variables = super().getVariables()
        if self.b is not None:
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
        # for f in self.f:
        #     variables.extend(f.quant_out.getQuantVariables())
        if self.b is not None:
            variables.extend(self.b.quant_out.getQuantVariables())
        # variables.extend(self.conv2d.mat_mul.quant_mul.getQuantVariables())
        # variables.extend(self.conv2d.mat_mul.quant_out.getQuantVariables())
        # variables.extend(self.conv2d.mat_mul.quant_sum.getQuantVariables())
        if self.b is not None:
            variables.extend(self.bias_add.quant_out.getQuantVariables())
        variables.extend(self.activation.getQuantVariables())
        variables.extend(self.quant_out.getQuantVariables())
        variables.extend(self.quant_in.getQuantVariables())
        return variables
        
    def build(self, input_shape):
        """build the variables of the layer
        
        called by TensorFlow before the first forward propagation.
        
        Parameters:
            input_shape (tuple of 4 ints):
                shape of the input tensor.
        """
        self.input_shape_ = input_shape
        input_channels = input_shape[3]
        self.f.build([*self.kernel_sizes, input_channels, 1])
        # for f in self.f:
        #     f.build([*self.kernel_sizes, 1, 1])
        if self.b is not None:
            self.b.build(input_channels)
        self.built = True
        
    def isQuantisizing(self):
        """is quantisizing
        
        test if the layer performs some quantisized operations.
        
        Returns:
            (bool): 
                true if an operation is quantisized.
        """
        return self.bias_add.isQuantisizing or self.quant_out.isQuantisizing() #or self.conv2d.isQuantisizing()

    @tf.function
    def call(self, inputs):
        """forward propagation
        
        apply the layer operation in the forward pass.
        
        Parameters:
            inputs (list of tensors): 
                list of all input tensors.
                
        Returns:
            (tensor): 
                output of the layer.
        """
        # [_, i_h, i_w, i_c] = inputs.shape
        # input_channels = tf.split(value=inputs, num_or_size_splits=i_c, axis=3)
        # input_channels_reshaped = [tf.reshape(channel, shape=[-1, i_h, i_w, 1]) for channel in input_channels]
        # spatialed_channels = [self.conv2d([input_channels_reshaped[i], self.f[i]()]) for i in range(i_c)]
        # tmp = tf.concat(spatialed_channels, axis=3)
        tmp = self.quant_in(inputs)
        tmp = tf.nn.depthwise_conv2d(tmp, self.f(), [1, *self.conv2d.strides, 1], self.conv2d.padding)

        if self.b is not None:
            tmp = self.bias_add([tmp, self.b()])
        if self.batch_norm != None:
            tmp = self.batch_norm(tmp)
        tmp = self.activation(tmp)
        return self.quant_out(tmp)

    def get_config(self):
        config = super().get_config()
        config["strides"] = self.strides
        config["padding"] = self.padding
        config["type"] = "mquat.Conv2DDepthwiseLayer"
        return config