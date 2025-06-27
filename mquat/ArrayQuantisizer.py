#######################################################################################################
#######################################################################################################
################################                                       ################################
###############################                M Q U A T                ###############################
###############################  (Modular Quantization Aware Training)  ###############################
################################                                       ################################
#######################################################################################################
#######################################################################################################
##                                                                                                   ##
## Datum: 03.09.2019                                                                                 ##
##                                                                                                   ##
## Version: 0.9                                                                                      ##
##                                                                                                   ##
## Contact:                                                                                          ##
##  Martin Hardieck hardieck@uni-kassel.de                                                           ##
##  Fabian Wagner   f.wagner8@gmx.de                                                                 ##
##                                                                                                   ##
## Copyright: All Rights Reserved                                                                    ##
##                                                                                                   ##
#######################################################################################################
#######################################################################################################

import numpy as np
import tensorflow as tf

from .Quantizer import Quantizer


# quantization to specified points
# uses the Basis-Class for initialisation of parameters if needed
class ArrayQuantizer(Quantizer):
    def __init__(self, name, q_array, leak_clip=0.001):
        min_value = np.amin(q_array)
        max_value = np.amax(q_array)
        super().__init__(name, min_value=min_value, max_value=max_value)
        self.q_array = q_array
        self.q_reshaped = np.reshape(q_array, [len(q_array), 1])
        self.leak_clip = leak_clip

    def buildGraph(self, x_tf):
        @tf.custom_gradient
        def quantisize_array_(x_tf):
            # reshape x to flat tensor
            x_tmp = tf.reshape(x_tf, [-1])
            # build a grid of abs differences between x and q
            # find the the index of the minimal distance
            abs_diff = tf.abs(x_tmp - self.q_reshaped)
            min_index = tf.argmin(abs_diff, axis=0)
            # get the corresponding quantisation value
            y_tmp = tf.gather(self.q_array, min_index)
            # reshape the result back to its oroginal form
            y = tf.reshape(y_tmp, tf.shape(x_tf))

            # define the gradient calculation
            def grad(dy):
                is_out_of_range = tf.logical_or(x_tf < self.min_value, x_tf > self.max_value)
                return tf.where(is_out_of_range, self.leak_clip * dy, dy)

            return y, grad

        return quantisize_array_(x_tf)
