# -*- coding: utf-8 -*-
from abc import ABC

import os
import re
import json
import logging
import inspect
from enum import Enum
from typing import List, Tuple
from difflib import SequenceMatcher

import mquat.FixedPointQuantizer
import tensorflow as tf
import numpy as np

from .QEnums import QuantizerLocation
from .Conv2DDepthwiseLayer import Conv2DDepthwiseLayer
from .Reflectable import Reflectable
from .Quantizer import Quantizer
from .Conv2DLayer import Conv2DLayer
from .DenseLayer import DenseLayer
from .ComplexLayer import ComplexLayer
from .Variable import Variable
from .QuantizerBase import QuantizerBase, DEFAULT_DATATYPE

def find_best_candidates(name, all_names):
    all_results = []
    for cur_name in all_names:
        s = SequenceMatcher(None, name, cur_name)
        all_results.append([cur_name, s.ratio()])
    all_results = sorted(all_results, key=lambda x: -x[1])
    all_results = [x for x in all_results]
    if len(all_results) > 3:
        all_results = all_results[0:3]
    return [name for name, score in all_results]

class DataRateInfo:
    def __init__(self, input_channels, input_rate, inputs_per_cycle, filters, stride, outputs_per_cycle, weight_groups, is_misc, input_shape, weight_shape):
        self.weight_shape = weight_shape
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.inputs_per_cycle = inputs_per_cycle
        self.input_rate = input_rate
        self.filters = filters
        self.stride = stride
        self.outputs_per_cycle = outputs_per_cycle
        self.weight_groups = weight_groups
        self.is_misc = is_misc
        if is_misc:
            self.KPU_count = "-"
            self.KPU_count_fullyp = "-"
            self.weight_groups_count = "-"
        else:
            self.weight_groups_count = len(weight_groups)
            self.KPU_count = int((input_channels * filters) * input_rate)
            self.KPU_count_fullyp = int(input_channels * filters)

    def __str__(self):
        return f"""
        input_channels = {self.input_channels}
        inputs_per_cycle = {self.inputs_per_cycle}
        input_rate = {self.input_rate}
        filters = {self.filters}
        stride = {self.stride}
        outputs_per_cycle = {self.outputs_per_cycle}
        weight_groups = {self.weight_groups}
        """

class QNetBaseModel(tf.keras.Model, ABC, Reflectable):
    """quantisation framework network base model

    the base class for all network models.

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
        **kwargs:
            keyword arguments of the keras model class.
    Atributes:
        _input_shape (list of ints):
            input shape of the model without the batch dimension
        _output_shape (list of ints):
            output shape of the model without the batch dimension
        _target_shape (list of ints):
            target shape (y_true shape) of the model without the batch dimension
        quant_in (Quantizer):
            model input quantisizer
            default: NON_QUANT
    """

    def __init__(self, name, input_shape, output_shape, target_shape, **kwargs):
        super().__init__(name=name, **kwargs)
        # renaming neccesary because of unchangeable attributes of the keras model class
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._target_shape = target_shape

    def _single_data_rate(self, layer, input_channels, from_layer, result_data, result_layer, data_dict, all_jumps):
        layer: mquat.ComplexLayer = layer
        if layer in result_layer:
            return
        if from_layer == None:
            current_inputs_per_cycle = input_channels
            current_input_channels = input_channels
        else:
            if from_layer not in data_dict:
                from_from_layer = None
                for f_layer, t_layer in all_jumps:
                    if t_layer == from_layer:
                        from_from_layer = f_layer
                        break
                self._single_data_rate(from_layer, input_channels, from_from_layer, result_data, result_layer, data_dict, all_jumps)
            current_inputs_per_cycle = data_dict[from_layer]["cips"]
            current_input_channels = data_dict[from_layer]["cic"]

        filters = layer.getFilterCount()
        if filters is None:
            filters = current_input_channels
        stride = layer.getStridesCount()

        current_input_rate = current_inputs_per_cycle / current_input_channels
        outputs_per_cycle = (current_input_rate * filters) / stride ** 2

        weight_groups = []
        if layer.getFilterCount() is not None:
            weight_group_size = int(current_input_channels / current_inputs_per_cycle)  # todo WICHTIG!
            for i in range(0, int(current_inputs_per_cycle)):
                start_at = i * weight_group_size
                end_at = (i + 1) * weight_group_size
                weight_groups.append(f"[{start_at}:{end_at}, :, :, :]   ")
        if hasattr(layer, "w"):
            weights = tf.shape(layer.w.var).numpy().tolist()
        elif hasattr(layer, "f"):
            weights = tf.shape(layer.f.var).numpy().tolist()
        else:
            weights = []
        data = DataRateInfo(input_channels=current_input_channels, input_rate=current_input_rate,
                            inputs_per_cycle=current_inputs_per_cycle, filters=filters, stride=stride,
                            outputs_per_cycle=outputs_per_cycle, weight_groups=weight_groups,
                            is_misc=layer.getFilterCount() is None, input_shape=layer.input_shape_,
                            weight_shape=weights)
        result_data.append(data)
        result_layer.append(layer)
        data_dict[layer] = {"cips": outputs_per_cycle, "cic": filters, "data": data}


    def _data_rates(self, input_channels:int, layer_list:List[ComplexLayer], special_jumps:List[Tuple[ComplexLayer, ComplexLayer]]=None):
        all_jumps = []
        last_layer = None
        if special_jumps is not None:
            all_jumps.extend(special_jumps)
        for layer in layer_list:
            all_jumps.append([last_layer, layer])
            last_layer = layer
        # print("all_jumps", all_jumps)
        result_data = []
        result_layer = []
        data_dict = {}
        for from_layer, to_layer in all_jumps:
            self._single_data_rate(to_layer, input_channels, from_layer, result_data, result_layer, data_dict, all_jumps)

        return zip(result_data, result_layer)


    def buildGraphs(self):
        """ build the internal TensorFlow graphs
        create new graphs for the model used in training(fit), prediction and
        evaluation.
        After modifying to the Quantizers it has to be called to apply the changes
        !!!important:!!!
        has to be called after model.compile. Model.compile otherwise deletes
        the build graphs
        """
        self.make_train_function(force=True)
        self.make_test_function(force=True)
        self.make_predict_function(force=True)

    def getLayers(self):
        """get all layers of the model

        Returns:
            (list of Layers):
                list of all layers in the model.
        """
        return self.layers

    def __setQuantizerMatMul__(self, op_var, op_regex, quantizer):
        """
        Sets quantizer for the respective operation op_var
        Args:
            op_var: operation which shall be checked
            op_regex: regular expression
            quantizer: Quantizer which shall be set

        Returns:
            A list of operations where the quantizer was set
        """

        summary = []

        if hasattr(op_var, 'mat_mul'):

            if re.fullmatch(op_regex, "mat_mul"):
                summary.append("mat_mul")
                op_var.mat_mul.quant_out = quantizer

            if hasattr(op_var.mat_mul, 'quant_mul'):
                if re.fullmatch(op_regex, "quant_mul"):
                    summary.append("quant_mul")
                    op_var.mat_mul.quant_mul = quantizer

            if hasattr(op_var.mat_mul, 'quant_sum'):
                if re.fullmatch(op_regex, "quant_sum"):
                    summary.append("quant_sum")
                    op_var.mat_mul.quant_sum = quantizer

        return summary

    def setQuantizer(self, layer_regex_list: list, op_regex_list: list, quantizer: Quantizer, debug=False):
        """
        Sets quantizer for the respective layers and operations of the current model
        Args:
            layer_regex_list: Layers which shall be quantized as list eg. [r".*conv1.*", ...]
            op_regex_list: Operations which shall be quantized as list [r".*w_var*", r".*b_var*", ...]
            quantizer: Quantizer which shall be set eg. FixedPointQuantizer
            debug: If True, a summary will be printed
        """

        summary = dict()
        # All layers which can be quantized
        quantizeable_layers = [layer for layer in self.getLayers() if hasattr(layer, "quant_out") and layer.trainable is True]

        # All layers which should be quantized
        filtered_model_layers = [layer for layer in quantizeable_layers for layer_regex in layer_regex_list if re.fullmatch(layer_regex, layer.name)]

        for layer in filtered_model_layers:
            summary[layer.name] = list()

            for op_regex in op_regex_list:
                for op in layer.getVariables():
                    if re.fullmatch(op_regex, op.name):
                        summary[layer.name].append(op.name)
                        op.quant_out = quantizer

                summary[layer.name].extend(self.__setQuantizerMatMul__(layer, op_regex, quantizer))

        if debug is True:
            for key in summary.keys():
                layer_ops = summary[key]
                print(key)
                for op in layer_ops:
                    print("\t--> " + op)

    def getVariables(self) -> List[Variable]:
        """get all variables like weights and biases(trainable parameters)

        Returns:
            (list of Variables):
                list of all variables in all layers of the model.
        """
        var_layers = [layer for layer in self.getLayers() if hasattr(layer, "getVariables")]
        return [var for var_layer in var_layers for var in var_layer.getVariables()]

    def getQuantizers(self, quant_location:QuantizerLocation=QuantizerLocation.EVERYWHERE):
        """get all quantizers

        Returns:
            (list of Variables):
                list of all variables in all layers of the model.
        """
        var_layers = [layer for layer in self.getLayers() if hasattr(layer, "getQuantizers")]
        return [var for var_layer in var_layers for var in var_layer.getQuantizers(quant_location)]

    def getQuantVariables(self):
        """get all quantizer variables

        Returns:
            (list of Variables):
                list of all variables in all layers of the model.
        """
        var_layers = [layer for layer in self.getLayers() if hasattr(layer, "getQuantVariables")]
        return [var for var_layer in var_layers for var in var_layer.getQuantVariables()]

    def getQuantHirachyLevels(self):
        quantizers = self.getQuantizers()
        all_levels = set()
        for quant in quantizers:
            if quant.name == "NON_QUANT":
                continue
            all_levels.add(quant.hirachy_value)
        all_levels = list(sorted(all_levels))
        return all_levels

    def getQuantizersByHierarchy(self, hirachy_filter_func):
        quantizers = self.getQuantizers()
        all_filtered = []
        for quant in quantizers:
            if quant.name == "NON_QUANT":
                continue
            if hirachy_filter_func(quant.hirachy_value):
                all_filtered.append(quant)
        return all_filtered

    def getVariablesValuesListNP(self, quantisize=None, fuse_batch_norm=False, return_named_dict=False):
        """get the values of the variables as numpy arrays

        Parameters:
            quantisize (None or True or False):
                 None (Default):
                     use internal active state of the quantisizers in the variables.
                 True:
                     force quantization of the variables.
                 False:
                     force no quantization of the variables.
            fuse_batch_norm (bool):
                if true the batch norm weights are fused with the coresponding
                conv2d/dense layer.
                !!! at the moment it its assumed that if the batch norm variables are quantisized that
                they are quantisized in the same way as the corresponding weights and biases
        Returns:
            (list of numpy arrays):
                list of variable values in all layers of the model.
        """
        if fuse_batch_norm == False:
            if return_named_dict:
                return {var.name: var.getValuesNP(quantisize) for var in self.getVariables()}
            return [var.getValuesNP(quantisize) for var in self.getVariables()]

        var_layers = [layer for layer in self.getLayers() if hasattr(layer, "getVariables")]
        var_values_list = []
        var_names_list = []
        for main_var_layer in var_layers:
            sub_var_layers = [main_var_layer]
            if isinstance(main_var_layer, ComplexLayer) and len(main_var_layer.getSubLayers()) > 0:
                sub_var_layers = main_var_layer.getSubLayers()
            for var_layer in sub_var_layers:
                var_list: List[Variable] = var_layer.getVariables()
                var_values = [tf.cast(var.getValuesNP(quantisize), var_layer.dtype) for var in var_list]
                var_names = [var.name for var in var_list]
                if isinstance(var_layer, Conv2DLayer) and var_layer.batch_norm != None:
                    beta, gamma, mean, var, f, b = var_values
                    beta_name, gamma_name, mean_name, var_name, f_name, b_name = var_names
                    epsilon = var_layer.batch_norm.variance_epsilon
                    f_new = gamma * f / tf.sqrt(var + epsilon)
                    b_new = gamma * (b - mean) / tf.sqrt(var + epsilon) + beta
                    f_new = var_layer.f.quant_out(f_new, quantisize=quantisize)
                    b_new = var_layer.b.quant_out(b_new, quantisize=quantisize)
                    var_values_list.extend([f_new, b_new])
                    var_names_list.extend([f_name, b_name])
                elif isinstance(var_layer, Conv2DDepthwiseLayer) and var_layer.batch_norm != None:
                    beta, gamma, mean, var, f, b = var_values
                    beta_name, gamma_name, mean_name, var_name, f_name, b_name = var_names

                    f_shaped = tf.transpose(f, [0, 1, 3, 2])

                    epsilon = var_layer.batch_norm.variance_epsilon
                    f_new = gamma * f_shaped / tf.sqrt(var + epsilon)
                    b_new = gamma * (b - mean) / tf.sqrt(var + epsilon) + beta
                    f_new = var_layer.f.quant_out(f_new, quantisize=quantisize)
                    b_new = var_layer.b.quant_out(b_new, quantisize=quantisize)
                    f_new = tf.transpose(f_new, [0, 1, 3, 2])
                    var_values_list.extend([f_new, b_new])
                    var_names_list.extend([f_name, b_name])
                elif isinstance(var_layer, DenseLayer) and var_layer.batch_norm != None:
                    beta, gamma, mean, var, w, b = var_values
                    beta_nane, gamma_name, mean_name, var_name, w_name, b_name = var_names
                    epsilon = var_layer.batch_norm.variance_epsilon
                    w_new = gamma * w / tf.sqrt(var + epsilon)
                    b_new = gamma * (b - mean) / tf.sqrt(var + epsilon) + beta
                    w_new = var_layer.w.quant_out(w_new, quantisize=quantisize)
                    b_new = var_layer.b.quant_out(b_new, quantisize=quantisize)
                    var_values_list.extend([w_new, b_new])
                    var_names_list.extend([w_name, b_name])
                else:
                    var_values_list.extend(var_values)
                    var_names_list.extend(var_names)
        result_dict = {name: value for name, value in zip(var_names_list, var_values_list)}
        # for key in result_dict:
        #     print(key, np.shape(result_dict[key]))
        if return_named_dict:
            return {name: value for name, value in zip(var_names_list, var_values_list)}
        return [var_values.numpy() for var_values in var_values_list]

    # def getVariablesValuesListNP(self, quantisize=False, fuse_batch_norm=False):
    #     """get the values of the variables as numpy arrays

    #     Parameters:
    #         quantisize (bool):
    #             if true quantisize the weights.
    #             default value is True (quantisize).
    #         fuse_batch_norm (bool):
    #             if true the batch norm weights are fused with the coresponding
    #             conv2d/dense layer.

    #     Returns:
    #         (list of numpy arrays):
    #             list of variable values in all layers of the model.
    #     """
    #     var_layers = [layer for layer in self.layers if hasattr(layer, "getVariables")]
    #     var_values_list = []

    #     # Skipping fusion of batch normalization
    #     if fuse_batch_norm is False:
    #         #var_values_list = [var.getValuesNP(quantisize) for var in self.getVariables()]

    #         # Change for debug test: Datatype missmatch when using fuse_batch_norm=False
    #         var_layers = [layer for layer in self.layers if hasattr(layer, "getVariables")]
    #         for var_layer in var_layers:
    #             var_list = var_layer.getVariables()
    #             var_values = [tf.cast(var.getValuesNP(quantisize=quantisize), var_layer.dtype) for var in var_list]
    #             var_values_list.extend(var_values)

    #     # Fuse batch normalization
    #     else:
    #         var_layers = [layer for layer in self.layers if hasattr(layer, "getVariables")]

    #         for var_layer in var_layers:
    #             var_list = var_layer.getVariables()

    #             if isinstance(var_layer, Conv2DLayer) and var_layer.batch_norm is not None:

    #                 # Getting Variable values without quantization
    #                 var_values = [tf.cast(var.getValuesNP(quantisize=False), var_layer.dtype) for var in var_list]
    #                 beta, gamma, mean, var, f, b = var_values
    #                 epsilon = 0.0000001

    #                 # Quantize fused values
    #                 f_new = gamma * f / tf.sqrt(var + epsilon)
    #                 b_new = gamma * (b - mean) / tf.sqrt(var + epsilon) + beta

    #                 if quantisize is True:
    #                     f_new = self.quant_in.quant(f_new)
    #                     b_new = self.quant_in.quant(b_new)

    #                 var_values_list.extend([f_new, b_new])

    #             elif isinstance(var_layer, DenseLayer) and var_layer.batch_norm is not None:

    #                 # Getting Variable values without quantization
    #                 var_values = [tf.cast(var.getValuesNP(quantisize=False), var_layer.dtype) for var in var_list]
    #                 beta, gamma, mean, var, w, b = var_values
    #                 epsilon = 0.0000001

    #                 # Quantize fused values
    #                 w_new = gamma * w / tf.sqrt(var + epsilon)
    #                 b_new = gamma * (b - mean) / tf.sqrt(var + epsilon) + beta

    #                 if quantisize is True:
    #                     w_new = self.quant_in.quant(w_new)
    #                     b_new = self.quant_in.quant(b_new)

    #                 var_values_list.extend([w_new, b_new])
    #             else:
    #                 # Getting quantized Variable values
    #                 var_values = [tf.cast(var.getValuesNP(quantisize=quantisize), var_layer.dtype) for var in var_list]
    #                 var_values_list.extend(var_values)

    #     return var_values_list

    def removeAllBatchNormLayers(self):
        """ removes all batch_norm layers. Currently only used in the PreTrained ResNet example.
        """
        var_layers = [layer for layer in self.layers if hasattr(layer, "getVariables")]
        print("var_layers", len(var_layers))
        for main_var_layer in var_layers:
            sub_var_layers = [main_var_layer]
            if isinstance(main_var_layer, ComplexLayer) and len(main_var_layer.getSubLayers()) > 0:
                sub_var_layers = main_var_layer.getSubLayers()
                main_var_layer.batch_norm = None
            for var_layer in sub_var_layers:
                var_layer.batch_norm = None

    def setVariablesValuesListNP(self, values_list_np):
        """ load the model variables from a list of numpy arrays.
        sequential loading in oder of the model layers list.

        Parameters:
            values_list_np(list of numpy arrays):
                load the model variables with the coresponding list entries.
        """
        for mq_var, values_np in zip(self.getVariables(), values_list_np):
            mq_var.var.assign(tf.cast(values_np, mq_var.var.dtype))

    def saveVariablesNPZ(self, file_path, quantisize=False, fuse_batch_norm=False):
        """save all variable values in a npz file(numpy compressed byte array list).
        sequential saveing in oder of the model layers list.

        Parameters:
            file_path (string):
                path to the savefile name (file extension .npz)
            quantisize (bool):
                if true quantisize the weights before saving
                default value is True (quantisize).
            fuse_batch_norm (bool):
                if true the batch norm weights are fused with the coresponding
                conv2d/dense layer.
        """
        # save model weights as bytes (by view) to keep tensorflow datatypes
        # that are not regular part of numpy datatypes.
        # wrap the values with an additional array dimension, to get a view in any case
        # (handle scalar values)
        if not file_path.endswith(".npz"):
            file_path = file_path + ".npz"
        byte_values_list_np = [np.array([tf.cast(values_np, tf.float32)]).view(np.byte) for values_np in
                               self.getVariablesValuesListNP(quantisize, fuse_batch_norm)]
        np.savez_compressed(file_path, *byte_values_list_np)
        self.saveQuantVariablesNPZ(os.path.splitext(file_path)[0] + "_qvars.npz")

    def loadVariablesNPZ(self, file_path, try_load_quantizers=True, verbose=False, filter=None):
        """load all variable values from a npz file(numpy compressed byte array list)
        sequential loading in oder of the model layers list.

        Parameters:
            file_path (string):
                path to the file (file extension .npz)
        """
        # load the values of the variables as byte arrays
        # for all values of the list:
        # get a view of the values as datatype of the variable, remove the first dimension and
        # assign it to the variable
        print("Loading Variables from", file_path)
        if not file_path.endswith(".npz"):
            file_path = file_path + ".npz"
        all_npz_entries = np.load(file_path)
        byte_keys_list_np = all_npz_entries.keys()
        all_variables = self.getVariables()
        if filter is not None:
            all_variables = [var for var in all_variables if filter(var)]
        if len(all_variables) != len(byte_keys_list_np):
            if verbose:
                print("All NPZ entries:")
                for key in all_npz_entries:
                    print(f"  key={key} shape={np.shape(all_npz_entries[key])}")
                print("All model variables:")
                for var in all_variables:
                    print(f"  name={var.name} shape={np.shape(var.getValuesNP())}")
            raise Exception(f"amount of npz weights ({len(byte_keys_list_np)}) do not match amount of model variables ({len(all_variables)})! " +
                            " Wrong npz file? Maybe you have (not) fused batch norm variables? Or maybe use the filter_vars parameter to filter out weights in the model. ",
                            file_path)
        for mq_var, np_key in zip(all_variables, byte_keys_list_np):
            # print("mq_var", mq_var.name, "np_key", np_key, "np_shape", np.shape(all_npz_entries[np_key]))
            try:
                value = all_npz_entries[np_key].view(tf.float32.as_numpy_dtype)[0]
                mq_var.var.assign(tf.cast(value, mq_var.var.dtype))
            except:
                value = all_npz_entries[np_key].view(tf.float16.as_numpy_dtype)[0]
                mq_var.var.assign(tf.cast(value, mq_var.var.dtype))

        if try_load_quantizers:
            quant_vars_filepath = os.path.splitext(file_path)[0] + "_qvars.npz"
            if os.path.exists(quant_vars_filepath):
                print(quant_vars_filepath, "was found. Now loading quantizer variables")
                self.loadQuantVariablesNPZ(quant_vars_filepath)
            else:
                print(quant_vars_filepath, "not found. Loading of quantizer variables skipped")


    def saveQuantVariablesNPZ(self, file_path):
        """save all quantizer variable values in a npz file(numpy compressed byte array list).
        sequential saveing in oder of the model layers list.

        Parameters:
            file_path (string):
                path to the savefile name (file extension .npz)
            quantisize (bool):
                if true quantisize the weights before saving
                default value is True (quantisize).
            fuse_batch_norm (bool):
                if true the batch norm weights are fused with the coresponding
                conv2d/dense layer.
        """
        # save model weights as bytes (by view) to keep tensorflow datatypes
        # that are not regular part of numpy datatypes.
        # wrap the values with an additional array dimension, to get a view in any case
        # (handle scalar values)
        quant_variables = self.getQuantVariables()
        byte_values_dict_np = {}
        for vars in quant_variables:
            byte_values_dict_np[vars.name] = np.array([tf.cast(vars.numpy(), tf.float32)]).view(np.byte)
        #byte_values_list_np = [np.array([vars.numpy()]).view(np.byte) for vars in quant_variables]
        np.savez_compressed(file_path, **byte_values_dict_np)


    def loadQuantVariablesNPZ(self, file_path):
        """load all quantizer variable values from a npz file(numpy compressed byte array list)
        sequential loading in oder of the model layers list.

        Parameters:
            file_path (string):
                path to the file (file extension .npz)
        """
        # load the values of the variables as byte arrays
        # for all values of the list:
        # get a view of the values as datatype of the variable, remove the first dimension and
        # assign it to the variable
        byte_values_map_np = np.load(file_path)
        all_variables = self.getQuantVariables()
        something_wrong = False
        # print("All keys in the quantizer variables npz file are", [x for x in byte_values_map_np.keys()])
        if len(all_variables) == 0:
            print("WARNING! NO QUANTIZATION DEFINED IN MODEL! CANNOT LOAD QUANTIZERS!!")
            something_wrong = True

        for vars in all_variables:
            if vars.name in byte_values_map_np:
                try:
                    value = byte_values_map_np[vars.name].view(tf.float32.as_numpy_dtype)[0]
                    vars.assign(tf.cast(value, vars.dtype))
                    # print("loading quantizer variable", vars.name, tf.cast(value, vars.dtype))
                except:
                    value = byte_values_map_np[vars.name].view(tf.float16.as_numpy_dtype)[0]
                    vars.assign(tf.cast(value, vars.dtype))
                    # print("loading quantizer variable", vars.name, tf.cast(value, vars.dtype))


            else:
                print(vars.name, "was not found! Could not load this value and assign it to the quantizer! Best candidates would be", find_best_candidates(vars.name, [name for name in byte_values_map_np]))
                something_wrong = True
        for name in byte_values_map_np:
            found = False
            for vars in all_variables:
                if vars.name == name:
                    found = True
                    break
            if not found:
                print("No quantizer was defined in the model with name", name, "best matches are", find_best_candidates(name, [vars.name for vars in all_variables]))
                something_wrong = True
        if something_wrong:
            print("# # # ALL NAMES IN THE CHECKPOINT # # # ")
            print([name for name in byte_values_map_np])
            print("# # # ALL NAMES IN THE MODEL # # # ")
            print([vars.name for vars in all_variables])

    def saveOptimizerNPZ(self, file_path):
        """ save the model optimizer weights to a list of numpy byte arrays
        that will be stored in a npz file.
        sequential saveing in oder of the optimizer weights list.

        file_path (string):
            path to the savefile name (file extension .npz)
        """
        # save model optimizer values as bytes (by view) to keep tensorflow datatypes
        # that are not regular part of numpy datatypes.
        # wrap the values with an additional array dimension, to get a view in any case
        # (handle scalar values)
        byte_values_list_np = [np.array([tf.cast(values, tf.float32)]).view(np.byte) for values in self.optimizer.get_weights()]
        np.savez_compressed(file_path, *byte_values_list_np)

    def loadOptimizerNPZ(self, file_path):
        """ load the model optimizer weights from a list of numpy byte arrays
        that are stored in a npz file.
        sequential loading in oder of the optimizer weights list.

        Parameters:
            file_path(string):
                path to the optimizer weights file (.npz)
        """
        # trick/hack to create/initilaize the optimizer weights to load them with
        # the contents of the npz file
        # first copy the model weights
        values_copies_np = [values_np.copy() for values_np in self.getVariablesValuesListNP()]
        # train a dummy batch to initialize the optimizer weights
        self.train_on_batch(tf.zeros([1, *self._input_shape]), tf.zeros([1, *self._target_shape]))
        # restore the model weights
        self.setVariablesValuesListNP(values_copies_np)
        # load the optimizer weights from the npz file (bytes)
        byte_values_list_np = np.load(file_path).values()
        values_list_np = []
        # for all values of the list:
        # get a view of the values as datatype of the optimizer weight,
        # remove the first dimension and assign it to the variable
        if len(byte_values_list_np) != len(self.optimizer.get_weights()):
            raise Exception("unequal amount of weights in optimizer npz file and used optimizer. Cannot load weigths for optimizer!")
        for optim_values_np, byte_values_np in zip(self.optimizer.get_weights(), byte_values_list_np):
            values_list_np.append(byte_values_np.view(optim_values_np.dtype)[0])
        self.optimizer.set_weights(values_list_np)

    def saveStructureCustomFormat(self, folder_path):
        """save the model structure in custom format

        Parameters:
            folder_path (string):
                path to the folder to save the structure data.
        """
        os.makedirs(folder_path, exist_ok=True)

        struct_file = open(os.path.join(folder_path, "layers.txt"), "w")
        for layer in self.layers:
            if hasattr(layer, "saveStructureCustomFormat"):
                layer.saveStructureCustomFormat(folder_path, struct_file)
        struct_file.close()

    def saveVariablesCustomFormat(self, folder_path, gz_compress=False, quantisize=False):
        """save the variables in custom format

        Parameters:
            folder_path (string):
                path to the folder to save all variable data.
            gz_compress (bool):
                if true gz compression will be used.
            quantisize (bool):
                if true the variables will be saved as quantisized values.
        """
        os.makedirs(folder_path, exist_ok=True)

        for variable in self.getVariables():
            variable.saveCustomFormat(folder_path, gz_compress, quantisize)

    def loadVariablesCustomFormat(self, folder_path,
                                  gz_compressed=False):  # TODO: Make PArameter equal to save funktion... gz_compress
        """load all variables from custom format

        Parameters: 
            folder_path (string):
                path to the folder where the variables are saved.
            gz_compressed (bool):
                must be true if the variables were saved compressed.
        """
        for variable in self.getVariables():
            variable.loadCustomFormat(folder_path, gz_compressed)

    def getPartialModel(self, end=1):
        """get a partial model of the entire model from input to end layer

        Args:
            end: Index of the last layer of the partial model. Default value is 1, then it returns the Input and the
                 first layer of the network

        Returns:
            (keras functional model): functional partial model of the entire model
        """

        # Getting the input layer
        inputs = tf.keras.layers.Input(self._input_shape)
        output = self.layers[0](inputs)

        # Getting all processing layers from the first processing layer to the last layer which is set by end
        for index in range(1, end):
            output = self.layers[index](output)

        partial_model = tf.keras.Model(inputs, output, name="Sub-{}".format(self.name))

        return partial_model

    def getWeightsAndBiases(self, quantize=False, fuse_batch_norm=False):
        """get all weights and biases from all network layers.
        Args:
            quantize: If true, the weight and bias data will be quantized
            fuse_batch_norm: If true, Batch-Normalization will be fused into weight and bias data
        Returns:
            all weights from all layers as list
        """
        weight_out = []
        bias_out = []

        for layer in self.layers:

            if "conv" in layer.name or "dense" in layer.name:

                var_list = layer.getVariables()
                weights = None
                biases = None

                if layer.batch_norm is not None:

                    # Getting all layer values from the specific layer
                    var_values = [tf.cast(var.getValuesNP(quantisize=False), layer.dtype) for var in var_list]

                    # If batchnorm is enabled the values beta, gamma, mean, var, w and b are given
                    beta, gamma, mean, var, w, b = var_values
                    epsilon = layer.batch_norm.variance_epsilon

                    # Fuse batch norm
                    if fuse_batch_norm is True:

                        weights = gamma * w / tf.sqrt(var + epsilon)
                        biases = gamma * (b - mean) / tf.sqrt(var + epsilon) + beta

                    # Skip batch norm fuse
                    else:
                        weights = w
                        biases = b

                else:
                    # Getting all variables from the neuronal network without quantization
                    weights, biases = [tf.cast(var.getValuesNP(quantisize=False), layer.dtype) for var in
                                       var_list]

                # Quantize weights and biases
                weights = weights.numpy() if quantize is False else self.quant_in.quant(weights).numpy()
                biases = biases.numpy() if quantize is False else self.quant_in.quant(biases).numpy()

                weight_data = {
                    "layer": layer.name,
                    "variables": weights,
                }

                bias_data = {
                    "layer": layer.name,
                    "variables": biases,
                }

                weight_out.append(weight_data)
                bias_out.append(bias_data)

        return weight_out, bias_out

    def getActivations(self, sample_data, quantize=False):
        """ get all activation outputs from all layers depending on the data input (sample_data)

        sample_data: Tensor
            Single data sample from the dataset which was used for training

        quantize: If true, the activations will be quantized

        Returns:
            all activations from all layers as list
        """

        outputs = []

        for layer_index in range(0, len(self.layers)):

            layer = self.layers[layer_index]

            if isinstance(layer, Conv2DLayer) or isinstance(layer, DenseLayer):
                partial_model = self.getPartialModel(end=layer_index)
                partial_model_out = partial_model([sample_data])[0]

                activations = partial_model_out.numpy() if quantize is False else self.quant_in.quant(
                    partial_model_out).numpy()

                activation_data = {
                    'layer': self.layers[layer_index].name,
                    'variables': activations,
                }

                outputs.append(activation_data)

        return outputs

    def apply_squant(self):
        for mq_var in self.getVariables():
            mq_var.applySQUANT()