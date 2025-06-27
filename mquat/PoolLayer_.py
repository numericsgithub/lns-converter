# # -*- coding: utf-8 -*-
#
# import tensorflow as tf
# from .OperationLayer import OperationLayer
# from .QuantizerBase import NON_QUANT
#
#
# class PoolLayer(OperationLayer):
#     """2d pooling layer that fixes the broken gradients when using quantization with max pooling.
#
#        Parameters:
#            name (string):
#                the name of the layer in the Tensorflow graph.
#            window_shape (2d int tuple):
#                (height, width) of the pooling window.
#            strides (2d int tuple):
#                (stride height, stride width) move of the window after each
#                pool operation.
#            pooling_type (string):
#                defines the pooling algorithm:
#                "AVG":
#                    average of all values of the window.
#                "MAX"
#                    maximum of all values of the window.
#            padding (string):
#                defines the padding algorithm:
#                "SAME":
#                    if the output should have the same size.
#                    if the window moves over the border of the input tensor zeros
#                    are assumed.
#                "VALID": if the window is not alowed to move over the borders
#                    of the input tensor.
#            dtype (tf.dtypes.DType):
#                datatype of the layer's operations.
#                default is float32.
#
#        Attributes:
#            window_shape (2d int tuple):
#                (height, width) of the pooling window
#            strides (2d int tuple):
#                (stride height, stride width) move of the window after each
#                pool operation.
#            pooling_type (string):
#                defines the pooling algorithm:
#                "AVG":
#                    average of all values of the window.
#                "MAX"
#                    maximum of all values of the window
#            padding (string):
#                defines the padding algorithm:
#                "SAME":
#                    if the output should have the same size.
#                    if the window moves over the border of the input tensor zeros
#                    are assumed.
#                "VALID":
#                    if the window is not alowed to move over the borders
#                    of the input tensor.
#        """
#
#     def __init__(self, name, window_shape, strides=None, pooling_type="MAX",
#                  padding="VALID", use_fixed=False, dtype=tf.dtypes.float32, debug=False):
#         super().__init__(name, "pool", dtype=dtype)
#         if strides == None:
#             strides = window_shape
#         self.pool_size = window_shape
#         self.strides = strides
#         self.pooling_type = pooling_type
#         self.padding = padding
#         self.debug = True#debug
#         self.use_fixed = use_fixed
#         if self.debug:
#             self.maxpool_count = tf.Variable(0.0, trainable=False)
#             self.maxpool_got_count = tf.Variable(0.0, trainable=False)
#
#     def printout(self):
#         tf.print(self.name, "MaxPoolOp got", tf.floor((self.maxpool_got_count / self.maxpool_count) * 10000) / 100,
#                  "collision rate! (",
#                  self.maxpool_count, "/", self.maxpool_got_count, ") ", self.maxpool_got_count - self.maxpool_count,
#                  "collided use_fixed=", self.use_fixed)
#
#     @tf.function(jit_compile=True)
#     def getmaxpoolgrad(self, pooled, dy, inputs):
#         # todo I think the line below is the first step to implement custom strides that differ from the window shape
#         # This is an example for window shape and stride = (2,2)
#         # repeated_max = tf.image.extract_patches(inputs, sizes=(1, 2, 2, 1), strides=(1, 2, 2, 1), rates=[1, 1, 1, 1], padding="VALID")
#
#         # Just repeat the max pooling result. So if the window [[1.0 2.0], [3.0, 3.0]] was max pooled to 3.0
#         # The result is just repeated to match the shape of the input like [[3.0, 3.0], [3.0, 3.0]]
#         # By doing this we have only one unique value per window. The value is the max pooled value.
#         repeated_max = tf.repeat(pooled, self.pool_size[0], axis=1)
#         repeated_max = tf.repeat(repeated_max, self.pool_size[1], axis=2)
#
#         repeated_dy = tf.repeat(dy, self.pool_size[0], axis=1)
#         repeated_dy = tf.repeat(repeated_dy, self.pool_size[1], axis=2)
#         # Now just match the input with the repeated_max.
#         #   So the example window [[1.0 2.0], [3.0, 3.0]] max pooled to 3.0
#         #   resulting in a repeated_max window of [[3.0, 3.0], [3.0, 3.0]]
#         #   Thereby resulting in [[0.0, 0.0], [1.0, 1.0]]
#         # Returning 1.0 in this context means that the gradient is 100% considered.
#         # Returning 0.0 means that the gradient is thrown out.
#         # The standard maxpooling (tf.nn.pool(...,"MAX)) would produce this window instead [[0.0, 0.0], [1.0, 0.0]]
#         # That would be wrong! It only considers the first max value and discards the rest.
#         result = tf.where(inputs == repeated_max, repeated_dy, 0.0)
#         return result
#
#     @tf.function(jit_compile=True)
#     def logmaxpool(self, inputs, tmp):
#         # todo I think the line below is the first step to implement custom strides that differ from the window shape
#         # This is an example for window shape and stride = (2,2)
#         # repeated_max = tf.image.extract_patches(inputs, sizes=(1, 2, 2, 1), strides=(1, 2, 2, 1), rates=[1, 1, 1, 1], padding="VALID")
#
#         # Just repeat the max pooling result. So if the window [[1.0 2.0], [3.0, 3.0]] was max pooled to 3.0
#         # The result is just repeated to match the shape of the input like [[3.0, 3.0], [3.0, 3.0]]
#         # By doing this we have only one unique value per window. The value is the max pooled value.
#         repeated_max = tf.repeat(tmp, self.pool_size[0], axis=1)
#         repeated_max = tf.repeat(repeated_max, self.pool_size[1], axis=2)
#         # Now just match the input with the repeated_max.
#         #   So the example window [[1.0 2.0], [3.0, 3.0]] max pooled to 3.0
#         #   resulting in a repeated_max window of [[3.0, 3.0], [3.0, 3.0]]
#         #   Thereby resulting in [[0.0, 0.0], [1.0, 1.0]]
#         # Returning 1.0 in this context means that the gradient is 100% considered.
#         # Returning 0.0 means that the gradient is thrown out.
#         # The standard maxpooling (tf.nn.pool(...,"MAX)) would produce this window instead [[0.0, 0.0], [1.0, 0.0]]
#         # That would be wrong! It only considers the first max value and discards the rest.
#         maxpool_count_cur = tf.cast(tf.cast(tf.size(tmp), tf.float32) - tf.reduce_sum(tf.where(tmp == 0.0, 1.0, 0.0)), tf.float32)
#         maxpool_got_count_cur = tf.reduce_sum(tf.where(tf.logical_and(inputs == repeated_max, inputs != 0.0), 1.0, 0.0))
#         self.maxpool_count.assign_add(maxpool_count_cur)
#         self.maxpool_got_count.assign_add(maxpool_got_count_cur)
#
#
#     def call(self, inputs):
#         """forward propagation
#
#         apply the pool operation in the forward pass
#
#         Parameters:
#             inputs (list):
#                 list of input tensors.
#
#         Returns:
#             (tensor):
#                 output of the layer.
#         """
#         inputs = self.quant_in(inputs)
#         if not self.use_fixed:
#             tmp = tf.nn.pool(inputs, self.pool_size, self.pooling_type, self.strides, self.padding)
#             if self.debug and self.pooling_type == "MAX":
#                 self.logmaxpool(inputs, tmp)
#             return self.quant_out(tmp)
#
#         else:
#             @tf.custom_gradient
#             def _quant(inputs):
#                 tmp = tf.nn.pool(inputs, self.pool_size, self.pooling_type, self.strides, self.padding)
#                 if self.debug and self.pooling_type == "MAX":
#                     self.logmaxpool(inputs, tmp)
#
#                 # define the gradient calculation
#                 def grad(dy):
#                     pooled = tf.nn.pool(inputs, self.pool_size, self.pooling_type, self.strides, self.padding)
#                     if self.pooling_type == "MAX":
#                         result = self.getmaxpoolgrad(pooled, dy, inputs)
#
#                     elif self.pooling_type == "AVG":
#                         # This is the normal backprop for AVG pooling
#                         result = tf.ones_like(inputs) / tf.cast(self.strides[0] * self.strides[1], dtype=tf.float32)
#                     else:
#                         raise Exception("Unknown pooling type" + str(self.pooling_type))
#                     tf.print("grads", self.name, result)
#                     return result
#                 return tmp, grad
#             return self.quant_out(_quant(inputs))
#
#
#
#     def saveStructureCustomFormat(self, folder_path, struct_file):
#         """save the layer structure in custom format
#
#         Parameters:
#             folder_path (string):
#                 path to the folder to save the layer data.
#             struct_file (file):
#                 file to save the structure data.
#         """
#         layer_file = self.writeCommonStructureCustomFormat(folder_path, struct_file)
#         layer_file.write("strides: " + " ".join(map(str, self.strides)) + "\n")
#         layer_file.write("pooling_type: " + self.pooling_type.upper() + "\n")
#         layer_file.write("padding: " + self.padding.upper() + "\n")
#         layer_file.close()
#
#     def get_config(self):
#         config = super().get_config()
#         config["input_shape"] = self.input_shape[1:]
#         config["output_shape"] = self.output_shape[1:]
#         config["padding"] = self.padding
#         config["strides"] = self.strides
#         config["pool_size"] = self.pool_size
#         config["data_format"] = "channels_last"
#         config["type"] = "mquat.PoolLayer"
#         return config