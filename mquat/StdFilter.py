import tensorflow as tf
from .Filter import Filter
from .QuantizerBase import DEFAULT_DATATYPE
from .Utilities import to_variable


class StdFilter(Filter):
    def __init__(self, name, std_keep_factor=3.0, dtype=DEFAULT_DATATYPE):
        super().__init__(name, dtype)
        self.active = True
        self.std_keep_factor = to_variable(std_keep_factor, tf.float32)


    @tf.function()
    def filterInputs(self, inputs, std_keep_factor):
        std_keep = tf.math.reduce_std(inputs) * std_keep_factor
        kept = tf.sort(inputs[tf.math.abs(inputs - tf.math.reduce_mean(inputs)) <= std_keep])
        #kept_s = tf.size(kept)
        #dropped = tf.sort(inputs[tf.math.abs(inputs - tf.math.reduce_mean(inputs)) > std_keep])
        #dropped_s = tf.size(dropped)
        # if self.debug:
        #     tf.print("FlexPointQuant: ", self.name,
        #              "kept", tf.round(100 * (kept_s / tf.size(inputs))), "%", kept_s, kept,
        #              "dropped", tf.round(100 * (dropped_s / tf.size(inputs))), "%", dropped_s, dropped, "input size is", tf.size(inputs))
        clip_min = tf.reduce_min(kept)
        clip_max = tf.reduce_max(kept)
        clipped = tf.clip_by_value(inputs, clip_min, clip_max)#tf.where(tf.math.abs(inputs - tf.math.reduce_mean(inputs)) <= std_keep,
                  #         inputs,
                  #         tf.clip_by_value(inputs, clip_min, clip_max))
        return clipped #kept#, clipped

    @tf.function()
    def quant(self, inputs, original_inputs):
        """quantisation function

        applies the quantization to the input.

        Parameters:
            inputs (list of tensors):
                list of all input tensors.

        Returns:
            (tensor):
                the output of layer.
        """
        inputs = tf.stop_gradient(self.filterInputs(inputs, self.std_keep_factor))
        return inputs