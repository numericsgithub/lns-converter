# import tensorflow as tf
# import numpy as np
# import gzip
# import os
#
# from .GlobalLoggingContext import GlobalLoggingContext
#
# logging_context = GlobalLoggingContext()
#
# class LoggingLayer(tf.keras.layers.Layer):
#     def __init__(self, name, append_to_log=True, dtype=tf.dtypes.float32):
#         super().__init__(False, name, dtype, False)
#         logging_context.all_loggers.append(self)
#         self.log_gz_file_path = os.path.join(logging_context.folder, name + ".gz")
#         self.append_to_log = append_to_log
#         self.step_counter = tf.Variable(0, trainable=False, dtype=tf.dtypes.int32)
#         self.num_steps_before_start = 0
#         self.num_steps_log = np.iinfo(np.int32).max
#         self.active = tf.Variable(False, trainable=False, dtype=tf.dtypes.bool)
#         self.log_file: gzip.GzipFile = None
#
#     def setStepParams(self, num_steps_before_start=None, num_steps_log=None):
#         if num_steps_before_start != None:
#             self.num_steps_before_start = num_steps_before_start
#         if num_steps_log != None:
#             self.num_steps_log = num_steps_log
#
#     def iterate_LogFile(self):
#         if self.log_file is not None:
#             raise Exception("Cannot iterate over log if the file is still open!")
#         with gzip.open(self.log_gz_file_path, "rb") as file:
#             for tensor_bytes_str in file.read().split(b"###")[:-1]:
#                 yield tf.io.parse_tensor(tensor_bytes_str, self.dtype)
#
#     def resetStepCounter(self):
#         self.step_counter.assign(0)
#
#     def activate(self):
#         self.active.assign(True)
#
#     def deactivate(self):
#         self.active.assign(False)
#
#     def close(self):
#         # print("CLOSING start")
#         self.flush()
#         if self.log_file is not None:
#             # print("CLOSING")
#             self.log_file.close()
#             self.log_file = None
#
#     def flush(self):
#         # print("FLUSHING start")
#         if self.log_file is not None:
#             # print("FLUSHING")
#             self.log_file.flush()
#
#     def writeTensor(self, inputs):
#         # tf.print("WRITING LOG", self.name)
#         tensor_bytes_str = tf.io.serialize_tensor(inputs).numpy()
#         if self.log_file is None:
#             if self.append_to_log:
#                 self.log_file = gzip.open(self.log_gz_file_path, "ab")
#             else:
#                 self.log_file = gzip.open(self.log_gz_file_path, "wb")
#         self.log_file.write(tensor_bytes_str + b"###")
#
#     def call(self, inputs):
#         if not self.active:
#             return inputs
#
#         self.step_counter.assign_add(1)
#
#         if self.step_counter <= self.num_steps_before_start:
#             return inputs
#
#         if self.step_counter > self.num_steps_before_start + self.num_steps_log:
#             return inputs
#
#         tf.py_function(func=self.writeTensor, inp=[inputs], Tout=[])
#         return inputs