# -*- coding: utf-8 -*-

import tensorflow as tf
import math
from .Reflectable import Reflectable
import os

DEFAULT_DATATYPE = tf.dtypes.float32
if 'MQUAT_DTYPE' in os.environ:
    if os.environ['MQUAT_DTYPE'] == "float16":
        DEFAULT_DATATYPE = tf.float16
    elif os.environ['MQUAT_DTYPE'] == "float32":
        DEFAULT_DATATYPE = tf.float32
    elif os.environ['MQUAT_DTYPE'] == "float64":
        DEFAULT_DATATYPE = tf.float64
    else:
        raise Exception("Unkown MQUAT_DTYPE set in env vars called \"" + os.environ['MQUAT_DTYPE'] + "\"")
    print("DEFAULT_DATATYPE set to", DEFAULT_DATATYPE)


class QuantizerBase(tf.keras.layers.Layer, Reflectable):
    """base class of the quantisizer layers
    
    the base class is needed to seperate the NON_QUANT from all 
    other quantisizers.
    
    Parameters:
        name (string):
            the name of the layer in the TensorFlow graph.
        dtype (tf.dtypes.DType):
            datatype of the layers's operations.
            default is float32.  
    
    Attributes:
        min_value (float):
            min value of the quantization range.
        max_value (float):
            max value of the quantization range.
    """
    def __init__(self, name, dtype=DEFAULT_DATATYPE, hirachy_value=None):
        super().__init__(name=name, trainable=False, dtype=dtype)
        self.min_value = tf.cast(-math.inf, DEFAULT_DATATYPE)
        self.max_value = tf.cast(math.inf, DEFAULT_DATATYPE)
        self.active = False
        self.hirachy_value = hirachy_value

    @property
    def doc_string(self):
        doc = "NON QUANT"
        return doc

    def create_logger_mapping(self):
        return None

    def call(self, input, quantisize=None):
        """forward propagation
        
        overwriten in subclasses.
        
        Parameters:
            input (tensor):
                input of the layer
            quantisize (None or True or False):
                argument is needed in the Quantizer Subclass.
        Returns:
            (tensor):
                returns the input tensor.
                no operation in QuantizerBase.
        """
        return input
    
    def activate(self):
        """enable the quantisizer
        
        overwriten in subclasses.
        """
        pass
    
    def deactivate(self):
        """disable the quantisizer
        
        overwriten in subclasses.
        """
        pass
    
    def isQuantisizing(self):
        """test if the quantisizer is quantisizing
        
        overwriten in subclasses.
        
        Returns:
            (bool): 
                return False.
                QuantizerBase not allowed to quantisize.
        """
        return False

    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        return variables

    def reset(self):
        pass

    def createConstraint(self, dtype):
        """create a network variable contraint
        
        the contraint prevents a Variable to be updated to values outside the 
        quantisation range in training.
        
        Parameters:
            dtype (tf.dtypes.DType):
                datatype of the weight/variable to constrain.
        Returns:
            (function):
                function that contraints a variable.
        """
        min_value = tf.cast(self.min_value, dtype)
        max_value = tf.cast(self.max_value, dtype)
        return lambda var: tf.clip_by_value(var, min_value, max_value)
    
    def createUniformInitializer(self):
        """create an uniform variable initializer
        
        uses the min an max values of the optimizer to prevent initializations
        outside the quantization range.
        
        Returns:
            (TensorFlow operation):
                function that initializes a variable.
        """
        return tf.random_uniform_initializer(self.min_value, self.max_value)
    
    def calculate_fan_in_out(self, shape):
        """calculate the fan in and fan out of a given weight shape
        
        Parameters:
            shape (list of int):
                shape of the weight/filter(variable)
        Returns:
            fan_in, fan_out (ints):
                fan factors used in the initialisation of the weight
        """
        # the fan_in and fan_out calculations are based on the implementation of the tf.contrib.layers.xavier_initializer 
        fan_in = shape[-2] if len(shape) > 1 else shape[-1]
        fan_out = shape[-1]
        for dim in shape[:-2]:
            fan_in *= dim
            fan_out *= dim
        return fan_in, fan_out
        
    def createUniformXavierInitializer(self, shape):
        """create a uniform xavier initializer.
        
        uses the min an max values of the optimizer to prevent initializations
        outside the quantization range. It is mainly used with TanH activation.
        
        Returns:
            (TensorFlow operation):
                function that initializes a variable.
        """
        
        fan_in, fan_out = self.calculate_fan_in_out(shape)
        # calculate the limit for the random_uniform initialisation
        limit = math.sqrt(6.0 / float(fan_in + fan_out))
        
        # limit the value range
        min_val = max(self.min_value, -limit)
        max_val = min(self.max_value,  limit)
        return tf.random_uniform_initializer(min_val, max_val)
    
    def createUniformHeInitializer(self, shape):
        """create a uniform he initializer.
        
        uses the min an max values of the optimizer to prevent initializations
        outside the quantization range. It is mainly used with ReLU activation.
        
        Returns:
            (TensorFlow operation):
                function that initializes a variable.
        """
        
        fan_in, fan_out = self.calculate_fan_in_out(shape)
        # calculate the limit for the random_uniform initialisation
        limit = math.sqrt(12.0 / float(fan_in + fan_out))
        
        # limit the value range
        min_val = max(self.min_value, -limit)
        max_val = min(self.max_value,  limit)
        return tf.random_uniform_initializer(min_val, max_val)

    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.QuantizerBase.NON_QUANT"
        config["quant_data"] = {"active": self.active}
        return config

# create one instance of the QuantiszerBase as NON_QUANT (non quantsisizer)
# the NON_QUANT is used in all classes as default
# simpliefies the usage of the framework (helps with initialisation of the variables)
NON_QUANT = QuantizerBase("NON_QUANT")