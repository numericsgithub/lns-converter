# -*- coding: utf-8 -*-
import random
import time

import numpy as np
import tensorflow as tf

from .Variable import Variable
from .Utilities import to_value, to_variable
from .RCCMQuantizer import RCCMQuantizer
import gc
#import matplotlib.pyplot as plt
from .QuantizerBase import DEFAULT_DATATYPE

import mquat.SQUANT.squant as sq


class AddQuantizer(RCCMQuantizer):
    """
    :param name: The name of the layer in the TensorFlow graph.
    :param adder_type: Specifies the RCCM to use. (2) for 2Add, (3) for 3Add, (4) for 4Add
    :param weights: The given weights to quantisize
    :param leak_clip: float
    :param dtype: tf.dtypes.DType
    """

    COEFFS_BASE_PATH = "coeffs/"
    rccm_coefficient_set_cache = {}
    __find_coeffs_locked = False

    def __init__(self, name, adder_type_name, pre_flex_bits=10, pre_keep_factor=0, find_coeff_set_py_func=None, leak_clip=0.0, dtype=DEFAULT_DATATYPE, custom_shift_overwrite=None, debug=False):
        if debug:
            print(f"Selected adder type for {name} is {adder_type_name}")
        self.adder_type_name = adder_type_name
        self.custom_shift_overwrite = custom_shift_overwrite
        self.__debug = debug
        self.__weight_transform_coeffs_shifted = None
        if adder_type_name not in AddQuantizer.rccm_coefficient_set_cache: # TODO Load all coeff sets so you can change them dynamically
            AddQuantizer.rccm_coefficient_set_cache[adder_type_name] = self.read_addfile(AddQuantizer.COEFFS_BASE_PATH + adder_type_name + ".txt", name)
            #AddQuantizer.rccm_coefficient_set_cache[to_value(adder_type, int)] = self.read_addfile("/home/fdai0217/coeffs/" + str(_adder_type) + "Add" + str(adder_w) + ".txt", name)
        coeff = tf.Variable(np.zeros([len(AddQuantizer.rccm_coefficient_set_cache[adder_type_name]) - 1, 1]), name=name + "_coeff", shape=[None, 1], trainable=False, dtype=dtype)
        self.coeff_shift = tf.Variable(name=name + "_coeff_shift", initial_value=0.0, trainable=False, dtype=dtype)
        self.find_coeff_set_py_func = find_coeff_set_py_func
        if self.find_coeff_set_py_func is None:
            self.find_coeff_set_py_func = self.find_coefficient_set_default_py_func
        super().__init__(name, coeff, self.adder_type_name, find_coeff_set_py_func=self.__find_coefficient_set_wrapper,
                         leak_clip=leak_clip, pre_flex_bits=pre_flex_bits, pre_keep_factor=pre_keep_factor, dtype=dtype)


    # def getVariables(self):
    #     """get all variables of the layer.
    #
    #     Returns:
    #         (list of Varaiables):
    #             list contains the weight and the bias Variable.
    #     """
    #     variables = super().getVariables()
    #     variables.extend([self.coeff, self.sample_counter]) # todo this is wrong! should be a mq.Variable
    #     return variables

    def reset(self):
        """
        Resets the Quantizer. The next inference triggers the coefficient search again.
        Returns:

        """
        pass


    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = super().getQuantVariables()
        variables.extend([self.coeff_shift])
        return variables

    def read_addfile(self, addfile_path: str, name :str):
        # reading the add file from disk
        # and converting each line:
        # "12 3 -4" -> np.array([12, 3, -4])
        # putting each np.array(...) in one list.
        # Also the longest len(np.array(...)) is searched for.
        if self.__debug:
            print("AddQuantizer: "+name+" reading the add file", addfile_path)
        unsorted_arrays = []
        start = time.time()
        line_max_count = 0
        with open(addfile_path) as file:
            while True:
                next_n_lines = file.readlines(10000)
                if not next_n_lines:
                    break
                while next_n_lines:
                    line = next_n_lines.pop()
                    if "|  " in line:
                        line = line[line.index("|  ") + 3:]
                        line = np.fromstring(line, dtype=np.float32, sep=" ")
                        unsorted_arrays.append(line)
                        if len(line) > line_max_count:
                            line_max_count = len(line)
                    del line
                del next_n_lines
        if self.__debug:
            print("AddQuantizer: "+name+" reading took ", round(time.time() - start, 3), "seconds")
        # sorting arrays by length
        # with the longest array found the arrays are all sorted
        # in a big list that groups arrays by length.
        # To keep it simple the index for any np.array(...) is its length.
        # But why? Simple, to convert each group in a 2D Array.
        # You know^^ Each 2D Array has the same amount of items in each row!
        # Thereby we need to sort them like that^^
        if self.__debug:
            print("AddQuantizer: "+name+" sorting arrays by length")
        sorted_numpy_arrays = []
        # create the big list and add an empty collection for each group
        for length in range(line_max_count + 1):
            sorted_numpy_arrays.append([])
        start = time.time()
        # put each np.array(...) in the correct group (determined by its length)
        for array in unsorted_arrays:
            sorted_numpy_arrays[line_max_count - len(array)].append(array)
        del unsorted_arrays

        # now with the np.array(...)s grouped by their length
        # we convert them to a huge 2D array :-)
        # and put it in reverse order. So the biggest coeffs sets are the first to look at
        for index in range(len(sorted_numpy_arrays)):
            sorted_numpy_arrays[index] = np.array(sorted_numpy_arrays[index])
        if self.__debug:
            print("AddQuantizer: " + name + " sorting took", round(time.time() - start, 3), "seconds")
            for i in range(len(sorted_numpy_arrays)):
                if len(sorted_numpy_arrays[i]) > 0:
                    print("AddQuantizer: " + name + " got", len(sorted_numpy_arrays[i]), "sets with",
                          len(sorted_numpy_arrays[i][0]), "coeffs each")
        # now if you want to get all the np.array(...)s with the biggests lengths
        # you can just sorted_numpy_arr ays[-1]
        return sorted_numpy_arrays

    def getCoeffs(self):
        return tf.cast(self.coeff.numpy(), self.dtype)

    def getCoeffsScaling(self):
        return tf.pow(2.0, -self.coeff_shift)

    def isNonUniform(self):
        return True

    def __weight_transform(self, weight_and_count):
        return tf.reduce_min(tf.abs(self.__weight_transform_coeffs_shifted - weight_and_count[0]) * weight_and_count[1],
                             axis=1)

    def __find_coefficient_set_wrapper(self, adder_type, weights):
        time.sleep(random.uniform(0, 1)) # TODO This is a totally legit way of synchronizing stuff! STOP JUDGING!!
        while AddQuantizer.__find_coeffs_locked:
            time.sleep(random.uniform(0, 1))
        AddQuantizer.__find_coeffs_locked = True
        try:
            result = tf.py_function(func=self.find_coeff_set_py_func, inp=[adder_type, weights], Tout=tf.float32)
        finally:
            AddQuantizer.__find_coeffs_locked = False
        return result


    def find_coefficient_set_default_py_func(self, adder_type, weights):
        """
        Finds the best coefficient set for the given weights for the given RCCM
        :param adder_type: Specifies the RCCM to use. (2) for 2Add, (3) for 3Add, (4) for 4Add
        :param weights: The given weights to quantisize

        :type adder_type: int
        :type weights: float[]

        :return: The most fitting coefficient set for the weights
        """
        #with tf.device('/cpu:0'):
        # Get coefficient sets of chosen RCCM
        all_coefficient_sets = AddQuantizer.rccm_coefficient_set_cache[adder_type.numpy().decode("ASCII")]
        start = time.time()
        # Declare Variables for Search
        most_fitting_coeff = []
        smallest_error = None

        # Find Point p
        try:
            point_p = np.log2(np.max(np.abs(weights)))
            if self.__debug:
                print("AddQuantizer: "+self.name+" point p is ", point_p)
        except:
            if self.__debug:
                print("AddQuantizer: " + self.name + " Error while finding min to find point p because log2 operation failed! Using default.")
            point_p = int(np.log2([0.1]))
        weights_unique, weights_count = np.unique(weights, return_counts=True)
        weights_sorted = np.stack([weights_unique, weights_count]).T
        weights_sorted = weights_sorted.tolist()
        weights_sorted.sort(key=lambda x: -x[1])
        if self.__debug:
            print("AddQuantizer: "+self.name+" amount of unique weights", len(weights_sorted), "amount of weights", np.size(weights),
              "[min;max]", "[" + str(np.min(weights)) + ";" + str(np.max(weights)) + "]", flush=True)
            print("AddQuantizer: " + self.name + " [unique weights, count] are", weights_sorted, flush=True)
            # fig = plt.figure()
            # ax = fig.add_axes([0.1,0.1,0.89,0.89])
            # plt.xticks(rotation=45)
            # ax.bar(weights_unique, weights_count, width=0.01)
            # fig.savefig("AddQuantizer_" + self.name + ".png")
            # plt.close(fig)

        # This part has to be rewritten by myself (Tobias Habermann)
        # I will do that soon. It should be implemented using tensorflow not numpy^^
        # Find the most fitting coefficient set

        # def try_other_stuff(weights, weights_quant, coeff_set):
        #     xsquant, results_abs_sum, results_sum = sq.adaptive_round(weights.numpy(), weights_quant.numpy(),
        #                                                               coeff_set.numpy(), isNonUniform=True)
        #     return results_abs_sum
        #
        # @tf.function(reduce_retracing=True, experimental_relax_shapes=True)
        # def try_stuff(coeff_set):
        #     # print("inner", tf.shape(coeff_set).numpy(), tf.shape(weights).numpy())
        #     weights_quant = self.quant_to_coeffs(weights, coeff_set)
        #     result = tf.py_function(try_other_stuff, inp=(weights, weights_quant, coeff_set), Tout=tf.float32)
        #     return result
        #
        # @tf.function(reduce_retracing=True, experimental_relax_shapes=True)
        # def my_map(*args, **kwargs):
        #     return tf.map_fn(*args, **kwargs)

        lowest_errors_index_list = []
        lowest_errors_coeffs_list = []
        lowest_errors_coeffs_shift_list = []

        for coeffs in all_coefficient_sets:
            if not coeffs.any():
                continue
            if smallest_error is not None and smallest_error == 0.0:
                break
            coeffs_ones = tf.ones([len(coeffs)], dtype=tf.float32)
            for points in range(-1, 2):
                if smallest_error is not None and smallest_error == 0.0:
                    break
                # Bits needed for pre-decimal places for each coeff_set
                # plus the current shift around point_p
                coeffs_bits_before = tf.experimental.numpy.log2(np.max(np.abs(coeffs), axis=1))

                # Subtract point p from this and convert the bits to fractions.
                # So you can shift the coeffs by multiplication
                # coeffs_shift_by_value = np.power(np.ones(len(coeffs_bits_before)) * 2,
                #                        np.floor(point_p - coeffs_bits_before) + 1)
                custom_shift = tf.floor(point_p - coeffs_bits_before + points) + 1
                if self.custom_shift_overwrite is not None:
                    custom_shift = tf.ones_like(custom_shift) * self.custom_shift_overwrite
                coeffs_shift_by_value = tf.pow(coeffs_ones * 2, custom_shift) # -9.0

                # Shift the coeffs by multiplication
                coeffs_shifted = tf.multiply(coeffs, tf.reshape(coeffs_shift_by_value, [len(coeffs), 1]))
                # coeffs_shifted = coeffs_shifted[0:100]

                self.__weight_transform_coeffs_shifted = coeffs_shifted
                errsum = tf.reduce_sum(tf.map_fn(self.__weight_transform, tf.constant(weights_sorted)), axis=0)
                # print("Original shape stuff", tf.shape(self.__weight_transform_coeffs_shifted).numpy(), tf.shape(weights_sorted).numpy(), tf.shape(errsum).numpy())

                lowest_errors_index_list_entry = tf.math.top_k(-errsum, tf.minimum(tf.shape(errsum)[0], 500))
                lowest_errors_index_list.append(-lowest_errors_index_list_entry.values.numpy())
                lowest_errors_coeffs_list.append(tf.gather(coeffs_shifted, lowest_errors_index_list_entry.indices).numpy())
                lowest_errors_coeffs_shift_list.append(tf.gather(custom_shift, lowest_errors_index_list_entry.indices).numpy())

                if smallest_error is None or tf.reduce_min(errsum) < smallest_error:
                    if self.__debug:
                        print("AddQuantizer: "+self.name+" found a better set with error", tf.reduce_min(errsum).numpy(), "and size", len(coeffs_shifted[tf.argmin(errsum)]), coeffs, flush=True)
                    smallest_error = tf.reduce_min(errsum)
                    most_fitting_coeff = coeffs_shifted[tf.argmin(errsum)]
                    most_fitting_coeff_shift = custom_shift[tf.argmin(errsum)]
            del coeffs
        if self.__debug:
            print("AddQuantizer: "+self.name+" most fitting set with error", smallest_error.numpy(), "shifted coeffs "+str(most_fitting_coeff_shift.numpy())+" are", most_fitting_coeff.numpy(), flush=True)

        end = time.time()
        if self.__debug:
            print("AddQuantizer: "+self.name+" Time to find coeffs", end - start, flush=True)

        # error_list = tf.reshape(np.concatenate(lowest_errors_index_list).ravel(), -1)
        # min_smallest_error_topkwise = -tf.math.top_k(tf.unique(-error_list).y, 2).values[-1].numpy()
        # min_smallest_error_factorwise = -tf.math.top_k(tf.unique(-error_list).y, 2).values[0].numpy() * 1.5
        # min_smallest_error = np.minimum(min_smallest_error_topkwise, min_smallest_error_factorwise)
        #
        # new_smallest_error = None
        # new_smallest_error_old_metric = None
        # print("most_fitting_coeff BEFORE squant", smallest_error.numpy(), most_fitting_coeff.numpy().tolist())
        # started = time.time()
        # validated_coeffset_counter = 0
        #
        # for i in range(len(lowest_errors_coeffs_list)):
        #
        #     min_smallest_error_indeces = lowest_errors_index_list[i] <= min_smallest_error
        #     lowest_errors_coeffs_list[i] = lowest_errors_coeffs_list[i][min_smallest_error_indeces]
        #     lowest_errors_coeffs_shift_list[i] = lowest_errors_coeffs_shift_list[i][min_smallest_error_indeces]
        #
        #     if len(lowest_errors_coeffs_list[i]) == 0:
        #         continue
        #
        #     new_errsum = my_map(try_stuff, lowest_errors_coeffs_list[i],
        #                            parallel_iterations=8, back_prop=False)
        #     validated_coeffset_counter += np.shape(np.unique(lowest_errors_coeffs_list[i], axis=0))[0] # todo does this work?
        #     if new_smallest_error is None or new_smallest_error > tf.reduce_min(new_errsum):
        #         new_smallest_error = tf.reduce_min(new_errsum)
        #         new_smallest_error_old_metric = lowest_errors_index_list[i][tf.argmin(new_errsum)]
        #         most_fitting_coeff = lowest_errors_coeffs_list[i][tf.argmin(new_errsum)]
        #         most_fitting_coeff_shift = lowest_errors_coeffs_shift_list[i][tf.argmin(new_errsum)]
        #         print("AddQuantizer: squant search with "+self.name+" found a better set with error", new_smallest_error.numpy(), flush=True)
        # print("most_fitting_coeff AFTER squant", smallest_error.numpy(), "to", new_smallest_error_old_metric, most_fitting_coeff.tolist())
        # print("Squant error calculation took", time.time() - started, "seconds and validated", validated_coeffset_counter, "coeff sets")

        self.coeff.assign(tf.cast(tf.reshape(most_fitting_coeff, [-1, 1]), DEFAULT_DATATYPE))
        self.coeff_shift.assign(tf.cast(tf.reshape(most_fitting_coeff_shift, ()), DEFAULT_DATATYPE))
        coeffs = np.sort(self.coeff.numpy())
        self.min_value = tf.cast(np.min(coeffs), DEFAULT_DATATYPE)
        self.max_value = tf.cast(np.max(coeffs), DEFAULT_DATATYPE)

        del all_coefficient_sets
        gc.collect()
        return 0