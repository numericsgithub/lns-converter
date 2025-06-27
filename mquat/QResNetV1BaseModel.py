# -*- coding: utf-8 -*-

import tensorflow as tf
from .QNetClassModel import QNetClassModel
from .Conv2DLayer import Conv2DLayer
from .AddLayer import AddLayer
from .ActivationLayer import ActivationLayer
from .QuantizerBase import DEFAULT_DATATYPE


class QResNetV1BaseModel(QNetClassModel):
    def __init__(self, name, input_shape, output_shape, target_shape, weight_decay=0.0001, dtype=DEFAULT_DATATYPE):
        super().__init__(name, input_shape, output_shape, target_shape, dtype)
        if weight_decay != None:
            self.weight_regularizer = tf.keras.regularizers.l2(weight_decay)
        else:
            self.weight_regularizer = None
        self.stack_list = []
        
    def createResidualBlock(self, name, channel_list, kernel_size_list, do_downsample,
                            kernel_initializer="he_normal", bias_initializer="zeros",
                            activation_fn=tf.nn.relu, do_batch_norm=True, 
                            dtype=DEFAULT_DATATYPE):
        downsample_stride = 2 if do_downsample == True else 1
        if len(kernel_size_list) == 2:
            stride_list = [downsample_stride, 1]
        else:
            stride_list = [1, downsample_stride, 1]
            
        conv_layer_list = []
        for index, [channels, kernel_size, stride] in enumerate(zip(channel_list, kernel_size_list, stride_list), 1):
            layer_activation_fn = activation_fn if index < len(channel_list) else tf.identity
            do_layer_batch_norm = do_batch_norm if index < len(channel_list) else False
            conv = Conv2DLayer(f"{name}_conv{index}", channels, [kernel_size, kernel_size],
                               strides=(stride, stride), padding="SAME",
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=self.weight_regularizer,
                               bias_initializer=bias_initializer,
                               activation_func=layer_activation_fn, 
                               do_batch_norm=do_layer_batch_norm, dtype=dtype)
            conv_layer_list.append(conv)
            setattr(self, conv.name, conv)
            
        if do_downsample == True or channel_list[0] != channel_list[-1]:
            shortcut_conv = Conv2DLayer(name + "_shortcut_conv", channel_list[-1], (1, 1),
                                        strides=[downsample_stride, downsample_stride], 
                                        padding="SAME",
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=self.weight_regularizer,
                                        bias_initializer=bias_initializer,
                                        activation_func=tf.identity, 
                                        do_batch_norm=False, dtype=dtype)
            setattr(self, shortcut_conv.name, shortcut_conv)
        else:
            shortcut_conv = None
            
        shortcut_add = AddLayer(f"{name}_shortcut_add_op", dtype)
        setattr(self, shortcut_add.name, shortcut_add)
                
        activation = ActivationLayer(f"{name}_activation_op", activation_fn, dtype)
        setattr(self, activation.name, activation)
                
        return [conv_layer_list, shortcut_conv, shortcut_add, activation]
    
    def createBlockStack(self, name, num_blocks, channel_list, kernel_size_list, do_downsample,
                         kernel_initializer="he_normal", bias_initializer="zeros",
                         activation_func=tf.nn.relu, do_batch_norm=True, 
                         dtype=DEFAULT_DATATYPE):
        block_stack = []
        for index in range(1, num_blocks+1):
            do_downsample_block = do_downsample == True and index == 1
            block = self.createResidualBlock(f"{name}_block{index}", channel_list, 
                                             kernel_size_list, do_downsample_block,  
                                             kernel_initializer, bias_initializer,
                                             activation_func, do_batch_norm, 
                                             dtype=DEFAULT_DATATYPE)
            block_stack.append(block)
        self.stack_list.append(block_stack)
    
    def callAllStacks(self, inputs):
        tmp = inputs
        for block_stack in self.stack_list:
            for block in block_stack:
                conv_layer_list, shortcut_conv, shortcut_add, activation = block
                x_tmp = tmp
                for conv_layer in conv_layer_list:
                    tmp = conv_layer(tmp)
                shortcut = shortcut_conv(x_tmp) if shortcut_conv != None else x_tmp
                tmp = shortcut_add([tmp, shortcut])
                tmp = activation(tmp)
        return tmp