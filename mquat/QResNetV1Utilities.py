# -*- coding: utf-8 -*-

import tensorflow as tf
from .Conv2DLayer import Conv2DLayer
from .AddLayer import AddLayer
from .ActivationLayer import ActivationLayer
from .QuantizerBase import DEFAULT_DATATYPE


def createResidualBlock(x, name, channel_list, kernel_size_list, do_downsample,
                        kernel_initializer="he_normal", bias_initializer="zeros",
                        activation_fn=tf.nn.relu, do_batch_norm=True,
                        weight_regularizer=None, dtype=DEFAULT_DATATYPE):
    
        downsample_stride = 2 if do_downsample == True else 1
        stride_list = [downsample_stride, 1] if len(kernel_size_list) == 2 else [1, downsample_stride, 1]
        
        tmp = x
        for index, [channels, kernel_size, stride] in enumerate(zip(channel_list, kernel_size_list, stride_list), 1):
            layer_activation_fn = activation_fn if index < len(channel_list) else tf.identity
            tmp = Conv2DLayer(f"{name}_conv{index}", channels, [kernel_size, kernel_size],
                              strides=[stride, stride], padding="SAME",
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=weight_regularizer,
                              bias_initializer=bias_initializer,
                              activation_func=layer_activation_fn, 
                              do_batch_norm=do_batch_norm, dtype=dtype)(tmp)
            
        if do_downsample == True or tmp.shape[-1] != x.shape[-1]:
            tmp_shortcut = Conv2DLayer(f"{name}_shortcut_conv",  tmp.shape[-1], [1, 1],
                                       strides=[downsample_stride, downsample_stride], 
                                       padding="SAME",
                                       kernel_initializer=kernel_initializer,
                                       kernel_regularizer=weight_regularizer,
                                       bias_initializer=bias_initializer,
                                       activation_func=tf.identity, 
                                       do_batch_norm=do_batch_norm, dtype=dtype)(x)
        else:
            tmp_shortcut = x
            
        tmp = AddLayer(f"{name}_shortcut_add_op", dtype)([tmp, tmp_shortcut])      
        return ActivationLayer(f"{name}_activation_op", activation_fn, dtype)(tmp)
    
    
def createBlockStack(x, name, num_blocks, channel_list, kernel_size_list, do_downsample,
                     kernel_initializer="he_normal", bias_initializer="zeros",
                     activation_func=tf.nn.relu, do_batch_norm=True, 
                     weight_regularizer=None, dtype=DEFAULT_DATATYPE):
    
    tmp = x
    for index in range(1, num_blocks+1):
        do_downsample_block = do_downsample == True and index == 1
        tmp = createResidualBlock(tmp, f"{name}_block{index}", channel_list, 
                                  kernel_size_list, do_downsample_block,  
                                  kernel_initializer, bias_initializer,
                                  activation_func, do_batch_norm, 
                                  weight_regularizer, dtype)   
    return tmp