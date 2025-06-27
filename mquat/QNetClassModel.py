# -*- coding: utf-8 -*-

import tensorflow as tf
from .QNetBaseModel import QNetBaseModel
from .QuantizerBase import NON_QUANT
from .QNetFunctionalModel import QNetFunctionalModel
from .QuantizerBase import DEFAULT_DATATYPE


class QNetClassModel(QNetBaseModel):
    """mquat class model
    
    use this class to build a model in the form of a keras class model.
    derive the class and create the layers in the constructor.
    overwrite the call method to implement the forward propagation through all layers.

    Parameters: 
        name (string):
            the name of the model in the TensorFlow graph.
        input_shape (list of ints):
            input shape of the model without the batch dimension
        output_shape (list of ints):
            output shape of the model without the batch dimension
        target_shape (list of ints):
            target shape (y_true shape) of the model without the batch dimension
        dtype (tf.dtypes.DType):
            datatype of the models's operations and weights/variables.
            default is float32.
            
    Attributes:
        quant_in (Quantizer):
            model input quantisizer
            default: NON_QUANT
    """
    def __init__(self, name, input_shape, output_shape, target_shape, dtype=DEFAULT_DATATYPE):
        super().__init__(name, input_shape, output_shape, target_shape, dtype=dtype)
        self.quant_in = NON_QUANT
        
    def build(self):
        """ build the model layers
        call build to set the layer input shapes and initialize the weights.
        the build method has to be called after the quantisizers are added to 
        the model to ensure that the weights are initialized to the right 
        quantisation ranges.
        """
        super().build([None, *self._input_shape])
        self.call(tf.keras.layers.Input(self._input_shape))
        
    def call(self, x):
        """forward propagation
        
        overwriten in the subclasses to implement the functionality.
        
        Parameters:
            x (tensor):
                the input tensor.
        """
        pass

    def getFunctionalModel(self):
         """get the model as a QNetFunctionalModel
                    
         the functional model holds a reference to the original model layers
         (the layers are not cloned)
         Returns: 
             (QNetFunctionalModel):
                 functional model with reference to the subclassed model
         """
         x = tf.keras.Input(self._input_shape, dtype=self.dtype)
         return QNetFunctionalModel(self.name, x, self.call(self.quant_in(x)), self._target_shape, self.dtype)