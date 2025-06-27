# -*- coding: utf-8 -*-

import tensorflow as tf
import mquat as mq
from mquat.QuantizerBase import DEFAULT_DATATYPE


class LeNet5_Like(mq.QNetBaseModel):
    def __init__(self, input_shape, output_shape, target_shape, do_batch_norm=False, dtype=DEFAULT_DATATYPE):
        super().__init__("LeNet5_Like", input_shape, output_shape, target_shape)
        self.conv1 = mq.Conv2DLayer("conv1", 8, (5,5), (1,1), padding="SAME", trainable_bias=True, activation_func=tf.nn.relu, do_batch_norm=False, dtype=dtype)
        self.pool1 = mq.PoolLayer("pool1_op", (2,2), (2,2), dtype=dtype)
        self.conv2 = mq.Conv2DLayer("conv2", 16, (5,5), (1,1), padding="SAME", trainable_bias=True, activation_func=tf.nn.relu, do_batch_norm=False, dtype=dtype)
        self.pool2 = mq.PoolLayer("pool2_op", (3,3), (3,3), dtype=dtype)
        self.flatten = mq.FlattenLayer("flatten_op", dtype)
        self.dense1 = mq.DenseLayer("dense1", output_shape[0], trainable_bias=True, activation_func=tf.nn.softmax, do_batch_norm=False,
                                    dtype=dtype)

    def call(self, inputs):
        tmp = inputs
        tmp = self.conv1(tmp)
        tmp = self.pool1(tmp)
        tmp = self.conv2(tmp)
        tmp = self.pool2(tmp)
        tmp = self.flatten(tmp)
        tmp = self.dense1(tmp)
        return tmp

    def data_rates(self, input_channels:int):
        layer_list = [
            self.conv1, self.pool1, self.conv2, self.pool2, self.dense1
        ]
        return self._data_rates(input_channels, layer_list)
