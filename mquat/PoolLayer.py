# -*- coding: utf-8 -*-

import tensorflow as tf

from .QEnums import QuantizerLocation
from .OperationLayer import OperationLayer
from .QuantizerBase import NON_QUANT
from .QuantizerBase import DEFAULT_DATATYPE


class PoolLayer(OperationLayer):
    """2d pooling layer

    Parameters:
        name (string):
            the name of the layer in the Tensorflow graph.
        window_shape (2d int tuple):
            (height, width) of the pooling window.
        strides (2d int tuple):
            (stride height, stride width) move of the window after each
            pool operation.
        pooling_type (string):
            defines the pooling algorithm:
            "AVG":
                average of all values of the window.
            "MAX"
                maximum of all values of the window.
        padding (string):
            defines the padding algorithm:
            "SAME":
                if the output should have the same size.
                if the window moves over the border of the input tensor zeros
                are assumed.
            "VALID": if the window is not alowed to move over the borders
                of the input tensor.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations.
            default is float32.

    Attributes:
        window_shape (2d int tuple):
            (height, width) of the pooling window
        strides (2d int tuple):
            (stride height, stride width) move of the window after each
            pool operation.
        pooling_type (string):
            defines the pooling algorithm:
            "AVG":
                average of all values of the window.
            "MAX"
                maximum of all values of the window
        padding (string):
            defines the padding algorithm:
            "SAME":
                if the output should have the same size.
                if the window moves over the border of the input tensor zeros
                are assumed.
            "VALID":
                if the window is not alowed to move over the borders
                of the input tensor.
    """

    def __init__(self, name, window_shape, strides=None, pooling_type="MAX",
                 padding="VALID", dtype=DEFAULT_DATATYPE):
        super().__init__(name, "pool", dtype=dtype)
        self.window_shape = window_shape
        self.strides = strides
        self.pooling_type = pooling_type
        self.padding = padding


    def getVariables(self):
        """get all variables of the layer

        Returns:
            (list of Variables):
                list contains filter and bias Variable.
        """
        variables = super().getVariables()
        return variables


    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend(self.quant_out.getQuantVariables())
        variables.extend(self.quant_in.getQuantVariables())
        return variables


    def getQuantizers(self, quant_location:QuantizerLocation=QuantizerLocation.EVERYWHERE):
        """get all quantizers of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        quantizers = []
        quantizers.append(self.quant_out)
        quantizers.append(self.quant_in)
        return quantizers

    def getFilterCount(self):
        return None

    def getStridesCount(self):
        return self.strides[0]

    def build(self, input_shape):
        self.input_shape_ = input_shape
        self.pool_input_shape = input_shape

    def call(self, inputs):
        """forward propagation

        apply the pool operation in the forward pass

        Parameters:
            inputs (list):
                list of input tensors.

        Returns:
            (tensor):
                output of the layer.
        """
        tmp = tf.nn.pool(inputs, self.window_shape, self.pooling_type, self.strides, self.padding)
        return self.quant_out(tmp)

    def saveStructureCustomFormat(self, folder_path, struct_file):
        """save the layer structure in custom format

        Parameters:
            folder_path (string):
                path to the folder to save the layer data.
            struct_file (file):
                file to save the structure data.
        """
        layer_file = self.writeCommonStructureCustomFormat(folder_path, struct_file)
        layer_file.write("strides: " + " ".join(map(str, self.strides)) + "\n")
        layer_file.write("pooling_type: " + self.pooling_type.upper() + "\n")
        layer_file.write("padding: " + self.padding.upper() + "\n")
        layer_file.close()

    def get_config(self):
        config = super().get_config()
        config["input_shape"] = self.input_shape[1:]
        config["output_shape"] = self.output_shape[1:]
        config["padding"] = self.padding
        config["strides"] = self.strides
        config["pool_size"] = self.window_shape  # TODO: should be renamed to pool_size!!!!
        config["data_format"] = "channels_last"
        config["type"] = "mquat.PoolLayer"
        return config