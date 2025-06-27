import tensorflow as tf
from .Filter import Filter
from .QuantizerBase import DEFAULT_DATATYPE
from .Utilities import to_variable


class PartialFilter(Filter):
    def __init__(self, name, quantized_portion=0.5, seed=42, dtype=DEFAULT_DATATYPE):
        super().__init__(name, dtype)
        self.active = True
        self.quantized_portion = to_variable(quantized_portion, tf.float32)
        self.seed = seed

    def build(self, input_shape):
        if input_shape[0] == None:
            input_shape = input_shape[1:]
        # limit = tf.cast(tf.size(tf.zeros(input_shape, dtype=self.dtype)), tf.float32) + 1.0
        # selection = tf.range(start=1.0, limit=limit) / limit
        # selection = tf.random.shuffle(selection, seed=self.seed)
        # selection = tf.cast(tf.reshape(selection, input_shape), tf.float32)
        # self.selection = selection.numpy()


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
        # limit = tf.stop_gradient(tf.cast(tf.size(tf.zeros_like(inputs, dtype=self.dtype)), tf.float32) + 1.0)
        # selection = tf.stop_gradient(tf.range(start=1.0, limit=limit) / limit)
        # selection = tf.stop_gradient(tf.random.shuffle(selection))
        # selection = tf.stop_gradient(tf.cast(tf.reshape(selection, tf.shape(inputs)), tf.float32))
        # tmp = tf.stop_gradient(tf.where(selection <= self.quantized_portion, inputs, original_inputs))
        ### tmp = tf.stop_gradient(tf.where(self.selection <= self.quantized_portion, inputs, original_inputs))


        selection = tf.random.uniform(tf.shape(inputs), minval=0.0, maxval=1.0)
        tmp = tf.stop_gradient(tf.where(selection <= self.quantized_portion, inputs, original_inputs))

        # tmp2 = tf.stop_gradient(tf.where(selection <= self.quantized_portion, 1.0, 0.0))
        # # tf.print(tf.reshape(tmp2, [-1]))
        # tf.print(tf.reduce_mean(tmp2))

        return tmp