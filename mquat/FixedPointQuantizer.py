# -*- coding: utf-8 -*-

import tensorflow as tf
from .LinearQuantizer import LinearQuantizer


class FixedPointQuantizer(LinearQuantizer):
    """fixed point quantisizer
    
    it is using a signed fixed point representation and is s based on the 
    LinearQuantizer.
    the quantisation range and number of steps is calculated by the 
    two bitsizes.
    """

    def __init__(self, name, channel_wise_scaling=False, scale_inputs=True):
        super(FixedPointQuantizer, self).__init__(name, channel_wise_scaling=channel_wise_scaling,
                                                     scale_inputs=scale_inputs)

    def setParams(self, bits_before=None, bits_after=None, leak_clip=None, scale=None):
        """ update/set quantisizer parameters
        
        important: after the parametes are set, call the model buildGraphs 
        method to apply the changes to the computation graph.
        
        Parameters:
            bits_before (int):
                bits before the comma
                if None (default): parameter is not modified.
            bits_after (int):
                bits after the comma
                if None (default): parameter is not modified.
            leak_clip (float):
                leak factor for backpropagation.
                when values are outside the quantsiation range (clipped to range)
                the gradient is multiplied by leak_clip
                if None (default): parameter is not modified.
        """
        if bits_before != None:
            self.bits_before = bits_before
        if bits_after != None:
            self.bits_after = bits_after
        if scale != None:
            self.scale.assign(scale)
    
        total_bits = self.bits_before + self.bits_after
        # q_steps   =  tf.pow(2.0, total_bits)
        # min_value = -tf.pow(2.0, bits_before - 1.0)
        # max_value =  tf.pow(2.0, bits_before - 1.0) - tf.pow(0.5, bits_after)
        q_steps   =  2 ** total_bits
        min_value = -2.0 ** (bits_before - 1)
        max_value =  2.0 ** (bits_before - 1) - 1.0 / (2 ** bits_after)
        super().setParams(min_value, max_value, q_steps, leak_clip)
        
    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.Quantizer.FixedPointQuantizer"
        config["quant_data"] = {"active": self.active,"wordsize": self.bits_before +  self.bits_after, "fraction_bits": self.bits_after,"shift": 0,  "scale": self.scale}
        return config