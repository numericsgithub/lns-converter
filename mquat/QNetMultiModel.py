# -*- coding: utf-8 -*-

from .QNetFunctionalModel import QNetFunctionalModel
from .QNetClassModel import QNetClassModel

import tensorflow as tf
from .QuantizerBase import DEFAULT_DATATYPE


class QNetMultiModel(QNetFunctionalModel):
    """multi submodel model
    
    the model contains submodels that share the same input. 
    each submodel has its own output.
    nested submodels are allowed and their outputs are flattened to a single output dict.
    subclass models are converted to functional models.

    Parameters: 
        name (string):
            the name of the model in the TensorFlow graph.
        sub_models (list of models):
            list of all submodels of types: QNetFunctionalModel or QNetClassModel or QNetMultiModel
        input_shape (list of ints):
            input shape of the model without the batch dimension
        output_shape (list of ints):
            output shape of the model without the batch dimension
        target_shape (list of ints):
            target shape (y_true shape) of the model without the batch dimension
        dtype (tf.dtypes.DType):
            datatype of the models's operations and weights/variables.
            default is float32.   
    Atributes:
        func_sub_models (list of QNetFunctionalModels):
            the functional submodels of the model.
    """
    def __init__(self, name, sub_models, input_shape, output_shape, target_shape,
                 dtype=DEFAULT_DATATYPE):
        func_sub_models = []
        x = tf.keras.Input(input_shape, dtype=dtype)
        y = []
        for sub_model in sub_models:
            # flatten nested submodels
            if isinstance(sub_model, QNetMultiModel):
                func_sub_models.extend(sub_model.getSubModels())
                for sub_sub_model in sub_model.getSubModels():
                    y.append(sub_sub_model(x))
            # convert ClassModels to FunctionalModels
            elif isinstance(sub_model, QNetClassModel):
                func_sub_model = sub_model.getFunctionalModel()
                func_sub_models.append(func_sub_model)
                y.append(func_sub_model(x))
            # add FunctionalModels wihtout conversion
            else:
                func_sub_models.append(sub_model)
                y.append(sub_model(x))
        super().__init__(name, x, y, target_shape, dtype)
        self.func_sub_models = func_sub_models
        
    def getSubModels(self):
        """get all submodels
                   
        Returns: 
            list of types: QNetFunctionalModel or QNetMultiModel
        """
        return self.func_sub_models
    
    def getLayers(self):
        """get all layers of the model
        
        Returns: 
            (list of Layers):
                flattened list of all layers in all submodels.
        """
        layers = []
        for sub_model in self.func_sub_models:
            layers.extend(sub_model.getLayers())
        return layers
    
    def getVariables(self):
        """get all variables(trainable parameters)
                
        Returns: 
            (list of Variables):
                list of all variables in all layers of the model.
        """
        variables = []
        for func_sub_model in self.func_sub_models:
            variables.extend(func_sub_model.getVariables())
        return variables
