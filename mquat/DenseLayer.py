# -*- coding: utf-8 -*-

import tensorflow as tf

from .QEnums import QuantizerLocation
from .ComplexLayer import ComplexLayer
from .Utilities import createLoggerEntry
from .Variable import Variable
from .MatMulOperationLayer import MatMulOperationLayer
from .AddLayer import AddLayer
from .QuantizerBase import DEFAULT_DATATYPE


class DenseLayer(ComplexLayer):
    """dense layer
    
    the dense layer first applies a matrix multiplication between weight matrix
    and the input, then the bias addition and at the end the activation.
    the matrix multiplication and the bias addition are OperationLayers that
    can be modified.
    
    Parameters: 
        name (string):
            the name of the layer in the TensorFlow graph.
        units (int):
            number outputs.
        weight_initializer (TensorFlow init operation):
            operation that initialisizes the weight variable.
        weight_regularizer (TensorFlow regularization operation):
            operation that regularizes the weight variable.
            None if deactivated.
        bias_initializer (TensorFlow init operation):
            operation that initialisizes the bias variable.
        bias_regularizer (TensorFlow regularization operation):
            operation that regularizes the bias variable
            None if deactivated.
        activation_func (function):
            actiavtion function of the layer.
            can be custom or from tf.nn.
        do_batch_norm (bool):
            true if batch nomalization is needed.
        trainable_bias (bool):
            to not use the bias at all use None.
            Else true or false to have trainable or non-trainable bias
        dtype (tf.dtypes.DType):
            datatype of the layer's operations and weights/variables.
            default is float32.

    Attributes:
        units (int):
            number outputs.
        w (Variable):
            the weight Variable of the matrix multiplication.
        b (Variable):
            the bias Variable.
        mat_mul (Conv2DOperationLayer):
            the matrix multiplication operation of the layer.
        bias_add (AddLayer):
            the bias addition operation of the layer.
    """
    def __init__(self, name, units,
                 weight_initializer="he_normal", weight_regularizer=None,
                 bias_initializer="zeros", bias_regularizer=None,
                 activation_func=tf.nn.relu, do_batch_norm=True, trainable_bias=True, dtype=DEFAULT_DATATYPE):
        super().__init__(name, activation_func, do_batch_norm, dtype=dtype)
        self.units = units
        self.w = Variable(name + "_w_var", weight_initializer, weight_regularizer, dtype=dtype)
        if trainable_bias is not None:
            self.b = Variable(name + "_b_var", bias_initializer, bias_regularizer, dtype=dtype, trainable=trainable_bias)
        else:
            self.b = None
        self.mat_mul = MatMulOperationLayer(name + "_matmul_op", dtype=dtype)
        self.bias_add = AddLayer(name + "_bias_add_op", dtype=dtype)

    GLOBAL_DEBUG_OUTPUT = False
    GLOBAL_DEBUG_OUTPUT_FILEPATH = ""

    def getFilterCount(self):
        return self.units

    def getStridesCount(self):
        return 1

    def getVariables(self):
        """get all variables of the layer.
        
        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = super().getVariables()
        variables.append(self.w)
        if self.b is not None:
            variables.append(self.b)
        return variables

    def create_logger_mapping(self):
        """
        Generates a dictionary holding the information about logging model
        Returns:

        """
        children = []
        children.append({"quant_in": self.quant_in.create_logger_mapping()})
        children.append({"weights": self.w.create_logger_mapping()})

        if self.b != None:
            children.append({"bias": self.b.create_logger_mapping()})
            #children.append(self.bias_add.create_logger_mapping())
        # if self.batch_norm != None:
        #     children.append(self.batch_norm.create_logger_mapping())
        # if hasattr(self.activation, "create_logger_mapping"):
        #     children.append(self.activation.create_logger_mapping())
        # children.append(self.quant_out.create_logger_mapping())
        return createLoggerEntry(
            type_base_name="layer",
            type_name="dense",
            name=self.name,
            params={"units": str(self.units)},
            children=children,
            loggers=[]
        )

    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend(self.w.quant_out.getQuantVariables())
        if self.b is not None:
            variables.extend(self.b.quant_out.getQuantVariables())
        variables.extend(self.mat_mul.quant_mul.getQuantVariables())
        variables.extend(self.mat_mul.quant_out.getQuantVariables())
        variables.extend(self.mat_mul.quant_sum.getQuantVariables())
        variables.extend(self.bias_add.quant_out.getQuantVariables())
        variables.extend(self.activation.getQuantVariables())
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
        if quant_location == QuantizerLocation.AT_WEIGHTS or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.w.quant_out)

        if quant_location == QuantizerLocation.AT_BIASES or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.b.quant_out)

        if quant_location == QuantizerLocation.AT_MULTIPLICATION or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.mat_mul.quant_mul)

        if quant_location == QuantizerLocation.AT_MULTIPLICATION or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.mat_mul.quant_out)

        if quant_location == QuantizerLocation.AT_AFTER_SUM or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.mat_mul.quant_sum)

        if quant_location == QuantizerLocation.AT_OUTPUT_ACTIVATIONS or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.bias_add.quant_out)

        if quant_location == QuantizerLocation.AT_OUTPUT_ACTIVATIONS or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.quant_out)

        if quant_location == QuantizerLocation.AT_INPUT_ACTIVATIONS or quant_location == QuantizerLocation.EVERYWHERE:
            quantizers.append(self.quant_in)
        return quantizers
        
    def build(self, input_shape):
        """build the variables of the layer.
        
        called by TensorFlow before the first forward propagation.
        
        Parameters:
            input_shape (tuple of 2 ints):
                shape of the input tensor.
        """
        self.input_shape_ = input_shape
        self.w.build([input_shape[1], self.units])
        if self.b != None:
            self.b.build(self.units)
        # mark the parent Layer class as built
        self.built = True

    def isQuantisizing(self):
        """is quantisizing
        
        test if the layer performs some quantisized operations.
        
        Returns:
            (bool):
                true if an operation is quantisized.
        """
        is_quantisizing = super().isQuantisizing()
        is_quantisizing = is_quantisizing or self.mat_mul.isQuantisizing()
        is_quantisizing = is_quantisizing or self.bias_add.isQuantisizing()
        return is_quantisizing

    def call(self, inputs, training):
        """forward propagation
        
        apply the layer operation in the forward pass.
        
        Parameters:
            inputs (list):
                list of all input tensors.
            training (bool):
                true: if in training mode.
                
        Returns:
            (tensor):
                the output of the layer.
        """
        tmp = self.quant_in(inputs)
        tmp = self.mat_mul([tmp, self.w()])
        if self.b != None:
            tmp = self.bias_add([tmp, self.b()])
        if self.batch_norm != None:
            tmp = self.batch_norm(tmp, training)
        tmp = self.activation(tmp)
        # tmp = self.quant_out(tmp)
        # test = tf.unique(tf.reshape(tmp, [-1])).y
        # tf.print("DENSE OUT", self.name, tf.size(test), tf.sort(test), summarize=-1)
        # tf.print("DENSE OUT FULL", self.name, tf.size(tmp), tmp, summarize=-1)
        # test = self.w()
        # tf.print("DENSE WEIGHTS FULL", self.name, tf.shape(test), test, summarize=-1)
        return self.quant_out(tmp)

    def saveStructureCustomFormat(self, folder_path, struct_file):
        """save the layer structure in custom format

        Parameters:
            folder_path (string):
                path to the folder to save the layer data.
            struct_file (file):
                file to save the structure data.
        """
        super().saveStructureCustomFormat(folder_path, struct_file)
        self.mat_mul.saveStructureCustomFormat(folder_path, struct_file)
        self.bias_add.saveStructureCustomFormat(folder_path, struct_file)

    def get_config(self):
        config = super().get_config()
        config3 = self.activation.get_config()
        config["type"] = "mquat.DenseLayer"
        config["input_shape"] = self.input_shape # with batch size in contrast to Conv layer where bath size is missing
        config["output_shape"] = self.output_shape # with batch size in contrast to Conv layer where bath size is missing
        config["units"] = self.units
        config["activation"] = self.activation.get_config()["activation"]
        config["data_format"] = "channels_last"
        config["quantize"] =  { "weight":    self.w.quant_out.get_config(), \
                                "bias":      self.b.quant_out.get_config(), \
                                "matmul_mul":self.mat_mul.quant_mul.get_config(), \
                                "matmul_sum":self.mat_mul.quant_sum.get_config(), \
                                "matmul_out":self.mat_mul.quant_out.get_config(), \
                                "bias_add":  self.bias_add.quant_out.get_config(),\
                                "out":       self.quant_out.get_config() \
                              }

        return config
