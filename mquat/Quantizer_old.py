# # -*- coding: utf-8 -*-
# import numpy as np
# import tensorflow as tf
# from .QuantizerBase import QuantizerBase
# from .QuantizerBase import DEFAULT_DATATYPE
# import matplotlib.pyplot as plt
#
#
# class Quantizer(QuantizerBase):
#     """base class for all quantisizers except the NON_QUANT
#
#     Parameters:
#         name (string):
#             the name of the layer in the TensorFlow graph.
#         dtype (tf.dtypes.DType):
#             datatype of the layers's operations.
#             default is float32.
#
#     Attributes:
#         active(bool):
#             default value is True (enabled).
#     """
#     def __init__(self, name, channel_wise_scaling=False, scale_inputs=False, dtype=DEFAULT_DATATYPE):
#         super().__init__(name, dtype)
#         # set the inital value to true (enabled)
#         self.active = True
#         self.scale = None
#         self.scale_error = None
#         self.zero_point = None
#         self.is_scaled = tf.Variable(not scale_inputs, trainable=False, name=name+"_is_scaled")
#         self.channel_wise_scaling = channel_wise_scaling
#
#     def build(self, input_shape):
#         if self.channel_wise_scaling:
#             scale_init = tf.ones(input_shape[-1])
#             zero_point_init = tf.zeros(input_shape[-1])
#         else:
#             scale_init = 1.0
#             zero_point_init = 0.0
#         self.scale = tf.Variable(scale_init, trainable=False, name=self.name+"_scale")
#         self.scale_error = tf.Variable(scale_init, trainable=False, name=self.name+"_scale_error")
#         self.zero_point = tf.Variable(zero_point_init, trainable=False, name=self.name+"_zero_point")
#
#     def getQuantVariables(self):
#         return [self.scale, self.scale_error, self.zero_point, self.is_scaled]
#
#     def setParams(self, min_value=None, max_value=None):
#         """ update/set quantisizer parameters
#
#         important: after the parametes are set, call the model buildGraphs
#         method to apply the changes to the computation graph.
#
#         Parameters:
#             min_value (float):
#                 min value of the quantization range.
#                 if None (default): parameter is not modified.
#             max_value (float):
#                 max value of the quantization range.
#                 if None (default): parameter is not modified.
#         """
#         if min_value != None:
#             self.min_value = min_value
#         if max_value != None:
#             self.max_value = max_value
#
#     def activate(self):
#         """enable the quantisizer
#         """
#         self.active = True
#
#     def deactivate(self):
#         """disable the quantisizer
#         """
#         self.active = False
#
#     def quant_forward(self, inputs):
#         return inputs
#
#     def isQuantisizing(self):
#         """test if the quantisizer is quantisizing
#
#         overwriten in subclasses.
#
#         Returns:
#             (bool):
#                 True if quantisizer is active.
#         """
#         return self.active
#
#     def plotScaling(self):
#         coeff = self.getCoeffs().numpy()
#
#
#
#     def findScale(self, inputs, scale_overwrite=None):
#         if self.is_scaled.numpy() == False:
#
#             tf.print(self.name + ": Searching for scale")
#             self.is_scaled.assign(True)
#             inputs = tf.cast(inputs, dtype=tf.float32)
#             def calc_error(scale):
#                 scale_cache = self.scale
#                 self.scale.assign(tf.cast(tf.ones_like(self.scale) * scale, self.scale.dtype))
#
#                 inputs_quanted, clipping_mask = self.quant_forward(inputs)
#                 not_clipping_mask = tf.logical_not(clipping_mask)
#
#                 quant_abs_error = tf.reduce_sum(tf.abs(inputs_quanted[not_clipping_mask] - inputs[not_clipping_mask]))
#                 clip_abs_error = tf.reduce_sum(tf.abs(inputs_quanted[clipping_mask] - inputs[clipping_mask]))
#
#                 quant_mse_error = inputs_quanted[not_clipping_mask] - inputs[not_clipping_mask]
#                 quant_mse_error = tf.reduce_sum(quant_mse_error * quant_mse_error)
#                 clip_mse_error = inputs_quanted[clipping_mask] - inputs[clipping_mask]
#                 clip_mse_error = tf.reduce_sum(clip_mse_error * clip_mse_error)
#
#                 self.scale.assign(scale_cache)
#                 return quant_abs_error, clip_abs_error, quant_mse_error, clip_mse_error
#
#             def calc_channel_wise_error(scale):
#                 scale_cache = self.scale
#                 self.scale.assign(tf.cast(scale, self.scale.dtype))
#
#                 inputs_quanted, clipping_mask = self.quant_forward(inputs)
#                 not_clipping_mask = tf.logical_not(clipping_mask)
#                 result = tf.reduce_sum(tf.abs(inputs_quanted - inputs), axis=[range(len(tf.shape(inputs)) - 1)])
#
#                 def f(inputs, mask):
#                     return tf.where(mask, inputs, 0.0)
#
#                 quant_abs_error = tf.reduce_sum(tf.abs(f(inputs_quanted, not_clipping_mask) - f(inputs, not_clipping_mask)), axis=[range(len(tf.shape(inputs)) - 1)])
#                 clip_abs_error = tf.reduce_sum(tf.abs(f(inputs_quanted, clipping_mask) - f(inputs, clipping_mask)), axis=[range(len(tf.shape(inputs)) - 1)])
#
#                 quant_mse_error = f(inputs_quanted,not_clipping_mask) - f(inputs,not_clipping_mask)
#                 quant_mse_error = tf.reduce_sum(quant_mse_error * quant_mse_error, axis=[range(len(tf.shape(inputs)) - 1)])
#                 clip_mse_error = f(inputs_quanted,clipping_mask) - f(inputs,clipping_mask)
#                 clip_mse_error = tf.reduce_sum(clip_mse_error * clip_mse_error, axis=[range(len(tf.shape(inputs)) - 1)])
#
#                 self.scale.assign(scale_cache)
#                 return quant_mse_error + clip_mse_error
#
#             # def calc_seek_scale(exp):
#             #     if exp > 0:
#             #         return 1.0 * tf.cast(tf.pow(2, exp), tf.float32)
#             #     else:
#             #         return 1.0 / tf.cast(tf.pow(2, -exp), tf.float32)
#
#             if self.channel_wise_scaling:
#                 self.zero_point.assign(tf.reduce_mean(inputs, axis=[range(len(tf.shape(inputs)) - 1)]))
#             else:
#                 self.zero_point.assign(tf.reduce_mean(inputs))
#
#             # self.zero_point.assign(0.0)
#
#             all_scales = []
#             all_errors = []
#
#             # Mit scale_plus=1.5 und * statt + -> OPT IS: ERROR:  11.859425 AT 29.192926025390626
#             # Mit scale_plus=0.5 und +         -> OPT IS: ERROR:  11.8373   AT 469.6
#             # Mit scale_plus=0.01 und +        -> OPT IS: ERROR:  11.846189 AT 29.340000000001446
#             # Mit neuen                        -> OPT IS: ERROR:  11.837467 AT tf.Tensor(469.265, shape=(), dtype=float32)
#             cur_scale_offset = 0.0
#             cur_scale_offset_error = 0.0
#             colors2 = ["gold", "orangered", "violet", "purple"]
#             colors = ["orange", "blue", "violet", "purple", "gold", "deepskyblue", "cyan", "pink"]
#             bases= [None, 2.0, 1.5, 1.25, 1.125]
#             cycles= [None, 20, 30, 30, 30]
#             biases = [None, 0.0, 1.0, 1.0, 1.0]
#             all_quant_abs_errors = []
#             all_clip_abs_errors = []
#             all_quant_mse_errors = []
#             all_clip_mse_errors = []
#             all_scales = []
#             for __ in range(1, 5):
#                 new_all_scales = []
#                 new_all_errors = []
#
#                 for _ in range(1, cycles[__]):
#                     change = tf.pow(bases[__], _/__) - biases[__]
#                     for scale_overwrite in range(210,700):
#                         if scale_overwrite:
#                             right_scalars = scale_overwrite
#                         else:
#                             right_scalars = cur_scale_offset + change
#                         quant_abs_error, clip_abs_error, quant_mse_error, clip_mse_error = calc_error(right_scalars)
#                         # right_error = quant_abs_error + clip_abs_error
#                         right_error = quant_mse_error + clip_mse_error
#                         new_all_scales.append(right_scalars)
#                         new_all_errors.append(right_error.numpy())
#                         all_quant_abs_errors.append(quant_abs_error)
#                         all_clip_abs_errors.append(clip_abs_error)
#                         all_quant_mse_errors.append(quant_mse_error)
#                         all_clip_mse_errors.append(clip_mse_error)
#
#                         # all_scales = []
#                         # all_scale_errors = []
#                         # for scale_overwrite in range(210, 700):
#                         #     if scale_overwrite:
#                         #         right_scalars = scale_overwrite
#                         #     else:
#                         #         right_scalars = cur_scale_offset + change
#                         #     quant_abs_error, clip_abs_error, quant_mse_error, clip_mse_error = calc_error(right_scalars)
#                         #     right_error = quant_abs_error + clip_abs_error
#                         #     # right_error = quant_mse_error + clip_mse_error
#                         #     new_all_scales.append(right_scalars)
#                         #     new_all_errors.append(right_error.numpy())
#                         #     all_quant_abs_errors.append(quant_abs_error)
#                         #     all_clip_abs_errors.append(clip_abs_error)
#                         #     all_quant_mse_errors.append(quant_mse_error)
#                         #     all_clip_mse_errors.append(clip_mse_error)
#                         #     all_scales.append(scale_overwrite)
#                         #     all_scale_errors.append(clip_mse_error.numpy())
#                         #     print("SCALE ABS TEST", scale_overwrite, clip_mse_error.numpy())
#                         # print("all_scales", all_scales)
#                         # print("all_scale_errors", all_scale_errors)
#
#                     if scale_overwrite:
#                         left_scalars = scale_overwrite
#                     else:
#                         left_scalars = cur_scale_offset - change
#                         left_scalars = tf.where(left_scalars <= 0.0, 0.0000001, left_scalars)
#                     quant_abs_error, clip_abs_error, quant_mse_error, clip_mse_error = calc_error(left_scalars)
#                     # left_error = quant_abs_error + clip_abs_error
#                     left_error = quant_mse_error + clip_mse_error
#                     new_all_scales.append(left_scalars)
#                     new_all_errors.append(left_error.numpy())
#                     all_quant_abs_errors.append(quant_abs_error)
#                     all_clip_abs_errors.append(clip_abs_error)
#                     all_quant_mse_errors.append(quant_mse_error)
#                     all_clip_mse_errors.append(clip_mse_error)
#                     if scale_overwrite:
#                         break
#                     # print("left_scalars", left_scalars.numpy().tolist(), "right_scalars", right_scalars.numpy().tolist(), cur_scale_offset, _/__, change)
#
#
#                 all_errors += new_all_errors
#                 all_scales += new_all_scales
#                 min_error_index = np.argmin(all_errors)
#                 cur_scale_offset = all_scales[min_error_index]
#                 cur_scale_offset_error = all_errors[min_error_index]
#                 print("cur tensor scale error", cur_scale_offset_error, "with scale", cur_scale_offset)
#                 if scale_overwrite:
#                     break
#
#             #     plt.scatter(new_all_scales, new_all_errors, label=f"Search step {__}", color=colors[__])
#             #
#             #     print("Tested scales at round", __, new_all_scales)
#             #
#             #
#             #     # plt.axhline(all_errors[min_error_index], color=colors[__], linestyle="-.")
#             #     plt.axvline(all_scales[min_error_index], color=colors[__], linestyle="-.")
#             #     print("OPT IS: ERROR: ", all_errors[min_error_index], "AT", all_scales[min_error_index])
#             #
#             # _, all_quant_abs_errors = zip(*sorted(zip(all_scales, all_quant_abs_errors), key=lambda x: x[0]))
#             # _, all_clip_abs_errors = zip(*sorted(zip(all_scales, all_clip_abs_errors), key=lambda x: x[0]))
#             # _, all_quant_mse_errors = zip(*sorted(zip(all_scales, all_quant_mse_errors), key=lambda x: x[0]))
#             # _, all_clip_mse_errors = zip(*sorted(zip(all_scales, all_clip_mse_errors), key=lambda x: x[0]))
#             # # plt.scatter(_, all_quant_abs_errors, label=f"quant abs errors ", color=colors2[0], marker="x")
#             # plt.plot(_, all_quant_abs_errors, label=f"quant abs errors ", color=colors2[0], marker="x")
#             # plt.plot(_, all_clip_abs_errors, label=f"clip abs errors ", color=colors2[1], marker="x")
#             # plt.plot(_, all_quant_mse_errors, label=f"quant mse errors ", color=colors2[2], marker="x")
#             # plt.plot(_, all_clip_mse_errors, label=f"clip mse errors ", color=colors2[3], marker="x")
#             #
#             # #plt.scatter(all_scales, all_errors, marker="x", color="green")
#             # min_error_index = np.argmin(all_errors)
#             # plt.axhline(all_errors[min_error_index], color="green", linestyle=":")
#             # plt.axvline(all_scales[min_error_index], color="green", linestyle=":")#
#             # print("OPT IS: ERROR: ", all_errors[min_error_index], "AT", all_scales[min_error_index])
#             # scale_and_errors = zip(all_scales, all_errors)
#             # scale_and_errors = sorted(scale_and_errors, key=lambda x: x[0])
#             # scale_and_errors = np.array(scale_and_errors)
#             #
#             #
#             # # plt.plot(scale_and_errors[:,0], scale_and_errors[:,1])
#             # plt.axvline(cur_scale_offset)
#             # plt.xlabel("Scale")
#             # plt.ylabel("Error (Absolute sum of errors")
#             # plt.title(f"Find a scaling for w for min err |Q(w)-w| {self.name}")
#             # plt.legend()
#             # plt.show()
#
#             # print(self.name +": SCALING CHOSEN!", cur_scale_offset)
#             self.scale.assign(tf.ones_like(self.scale) * cur_scale_offset)
#             self.scale_error.assign(cur_scale_offset_error)
#             if scale_overwrite:
#                 return 0.0
#
#             if self.channel_wise_scaling:
#
#                 # mymax = tf.reduce_max(inputs, axis=[range(len(tf.shape(inputs)) - 1)])
#                 cur_scale_offset = tf.ones_like(self.scale) * self.scale # enforce that it is a tensor and not a tf.variable
#                 cur_scale_offset_error = calc_channel_wise_error(cur_scale_offset)
#                 print("starting with cur_scale_offset_error", tf.reduce_sum(cur_scale_offset_error).numpy())
#                 bases = [None, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.03125]
#                 cycles = [None, 20, 30, 50, 60, 70, 80, 90]
#
#                 cycles = [None, 20, 25, 30, 40, 50, 40]
#                 biases = [None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#                 for __ in range(1, 7):
#                     print("AT INDEX", __)
#                     for _ in range(1, cycles[__]): # 9.9886055 tf.Tensor(12.928808, shape=(), dtype=float32)
#                         change = tf.pow(bases[__], _/__) - biases[__]
#                         left_scalars = cur_scale_offset - change
#                         left_scalars = tf.where(left_scalars <= 0.0, 0.0000001, left_scalars)
#                         left_error = calc_channel_wise_error(left_scalars)
#                         cur_scale_offset = tf.where(left_error < cur_scale_offset_error, left_scalars, cur_scale_offset)
#                         cur_scale_offset_error = tf.where(left_error < cur_scale_offset_error, left_error, cur_scale_offset_error)
#
#                         right_scalars = cur_scale_offset + change
#                         right_error = calc_channel_wise_error(right_scalars)
#                         cur_scale_offset = tf.where(right_error < cur_scale_offset_error, right_scalars, cur_scale_offset)
#                         cur_scale_offset_error = tf.where(right_error < cur_scale_offset_error, right_error, cur_scale_offset_error)
#                         print(_, "cur channel wise scale error", tf.reduce_sum(cur_scale_offset_error).numpy(), "with avg scale", tf.reduce_mean(cur_scale_offset).numpy())
#
#                 #
#                 # cur_scale_offset = tf.ones_like(self.scale) * cur_scale_offset
#                 # cur_scale_plus = tf.ones_like(self.scale) * 1.0
#                 # for _ in range(40):
#                 #     cur_error = calc_channel_wise_error(cur_scale)
#                 #     all_scales.append(cur_scale.numpy())
#                 #     all_errors.append(cur_error.numpy())
#                 #     # print(self.name + " current scale", cur_scale, all_errors[-1])
#                 #
#                 #     right_error = calc_channel_wise_error(cur_scale + cur_scale_plus)
#                 #     all_scales.append((cur_scale + cur_scale_plus).numpy())
#                 #     all_errors.append(right_error.numpy())
#                 #
#                 #     left_error = calc_channel_wise_error(cur_scale - cur_scale_plus)
#                 #     all_scales.append((cur_scale - cur_scale_plus).numpy())
#                 #     all_errors.append(left_error.numpy())
#                 #
#                 #     print(self.name + " current (channel wise) error", tf.reduce_sum(tf.abs(all_errors[-1])))
#                 #
#                 #     # if right_error < cur_error:
#                 #     #     cur_scale += cur_scale_plus
#                 #     # if left_error < cur_error:
#                 #     #     cur_scale -= cur_scale_plus
#                 #     cur_scale = tf.where(right_error < cur_error, cur_scale + cur_scale_plus, cur_scale)
#                 #     cur_scale = tf.where(left_error < cur_error, cur_scale - cur_scale_plus, cur_scale)
#                 #
#                 #     # if cur_error < min(right_error, left_error):
#                 #     #     cur_scale_plus = cur_scale_plus * 0.5
#                 #     # else:
#                 #     #     cur_scale_plus = cur_scale_plus * 1.25
#                 #     cur_scale_plus = tf.where(cur_error < tf.minimum(right_error, left_error),
#                 #                               cur_scale_plus * 0.5, cur_scale_plus * 2.0)
#
#                 self.scale.assign(tf.cast(cur_scale_offset, dtype=self.scale.dtype))
#             # exit(0)
#             # self.zero_point.assign(tf.reduce_mean(inputs))
#             # plt.scatter(all_scales, all_errors)
#             # plt.show()
#             # last_was_scaled_bigger = True
#             # last_error = calc_error()
#             # self.scale.assign(self.scale * 2.0)
#             # cur_error = calc_error()
#             # # seek position.
#             # while cur_error < last_error: # The scale will be to right of the optimum scale. scale > optimum
#             #     last_error = cur_error
#             #     self.scale.assign(self.scale * 2.0)
#             #     cur_error = calc_error()
#             #
#         return 0.0
#
#
#
#
#     def call(self, inputs, quantisize=None):
#         """forward propagation
#
#         if the quantisizer is active it calls the quant function on the input
#         and returns the result else it returns the input.
#
#         Parameters:
#             inputs (list of tensors):
#                 list of all input tensors.
#             quantisize (None or True or False):
#                 None (Default):
#                     use internal active state of the quantisizer.
#                 True:
#                     force quantization of the input (ignore the actual active state).
#                 False:
#                     force direct return of the input (ignore the actual active state).
#         Returns:
#             (tensor):
#                 quantisized imput.
#         """
#         if quantisize == True:
#             return self.quant(inputs)
#         elif quantisize == False:
#             return inputs
#         elif self.isQuantisizing():
#             return self.quant(inputs)
#         else:
#             return inputs
#
#     def quant(self, inputs):
#         """quantisation function
#
#         overwriten in the actual quantisizer.
#
#         Parameters:
#             inputs (list):
#                 list of all input tensors.
#
#         Returns:
#             (tensor):
#                 quantisized input.
#         """
#         return inputs
#
#     def get_config(self):
#         config = super().get_config()
#         config["type"] = "mquat.Quantizer"
#         config["quant_data"] = {"active": self.active}
#         return config