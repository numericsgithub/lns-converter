# -*- coding: utf-8 -*-

import tensorflow as tf
from .FlexPointQuantizer import FlexPointQuantizer


class MetaAutofilter:

    def __init__(self, flexpoint: FlexPointQuantizer):
        self.meta_initializer = None
        self.flexpoint = flexpoint

    def initial(self, inputs, training, meta_call_func):
        self.flexpoint.deactivate()
        fp32_output = meta_call_func(inputs, training)
        self.flexpoint.activate()
        quant_output = meta_call_func(inputs, training)
        abs_error = tf.abs(fp32_output- quant_output)
        tf.print("min, max error of quantization", tf.reduce_min(abs_error), tf.reduce_max(abs_error))
        return 0