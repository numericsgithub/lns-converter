import tensorflow as tf
from .QuantizerBase import QuantizerBase, DEFAULT_DATATYPE


class Filter(QuantizerBase):
    def __init__(self, name, dtype=DEFAULT_DATATYPE):
        super().__init__(name, dtype)
        self.active = True

    def create_logger_mapping(self):
        return {"name": self.name,
                "params": {},
                "children": [],
                "loggers": [ ("log_after_filter", self.log_after_quantization.name) ]}

    def build(self, input_shape):
        pass

    def getQuantVariables(self):
        return []

    def setParams(self, min_value=None, max_value=None):
        """ update/set quantisizer parameters

        important: after the parametes are set, call the model buildGraphs
        method to apply the changes to the computation graph.

        Parameters:
            min_value (float):
                min value of the quantization range.
                if None (default): parameter is not modified.
            max_value (float):
                max value of the quantization range.
                if None (default): parameter is not modified.
        """
        if min_value != None:
            self.min_value = min_value
        if max_value != None:
            self.max_value = max_value

    def activate(self):
        """enable the quantisizer
        """
        self.active = True

    def deactivate(self):
        """disable the quantisizer
        """
        self.active = False

    def quant_forward(self, inputs):
        return inputs

    def isQuantisizing(self):
        """test if the quantisizer is quantisizing

        overwriten in subclasses.

        Returns:
            (bool):
                True if quantisizer is active.
        """
        return self.active

    def call(self, inputs, quantisize=None):
        """forward propagation

        if the quantisizer is active it calls the quant function on the input
        and returns the result else it returns the input.

        Parameters:
            inputs (list of tensors):
                list of all input tensors.
            quantisize (None or True or False):
                None (Default):
                    use internal active state of the quantisizer.
                True:
                    force quantization of the input (ignore the actual active state).
                False:
                    force direct return of the input (ignore the actual active state).
        Returns:
            (tensor):
                quantisized imput.
        """
        if quantisize == True or self.isQuantisizing():
            result = self.quant(inputs)
            # self.log_after_quantization(result)
            return result
        return inputs

    def call(self, inputs, original_inputs, quantisize=None):
        """forward propagation

        if the quantisizer is active it calls the quant function on the input
        and returns the result else it returns the input.

        Parameters:
            inputs (list of tensors):
                list of all input tensors.
            quantisize (None or True or False):
                None (Default):
                    use internal active state of the quantisizer.
                True:
                    force quantization of the input (ignore the actual active state).
                False:
                    force direct return of the input (ignore the actual active state).
        Returns:
            (tensor):
                quantisized imput.
        """
        if quantisize == True or self.isQuantisizing():
            result = self.quant(inputs, original_inputs)
            # self.log_after_quantization(result)
            return result
        return inputs

    def quant(self, inputs, original_inputs):
        """quantisation function

        overwriten in the actual quantisizer.

        Parameters:
            inputs (list):
                list of all input tensors.

        Returns:
            (tensor):
                quantisized input.
        """
        return inputs

    def get_config(self):
        config = super().get_config()
        config["type"] = "mquat.Filter"
        config["quant_data"] = {"active": self.active}
        return config

# def create_filter_via_Std(name="std_filter", std_keep_factor=2.5):
#     def __filter(inputs):
#         std_keep = tf.math.reduce_std(inputs) * std_keep_factor
#         kept = tf.sort(inputs[tf.math.abs(inputs - tf.math.reduce_mean(inputs)) <= std_keep])
#         kept_s = tf.size(kept)
#         dropped = tf.sort(inputs[tf.math.abs(inputs - tf.math.reduce_mean(inputs)) > std_keep])
#         dropped_s = tf.size(dropped)
#         # if __filter.__debug:
#         #     tf.print("FlexPointQuant: ", __filter.name,
#         #              "kept", tf.round(100 * (kept_s / tf.size(inputs))), "%", kept_s, kept,
#         #              "dropped", tf.round(100 * (dropped_s / tf.size(inputs))), "%", dropped_s, dropped)
#         return kept
#     __filter.__name__ = name
#     return __filter
#
#
# # def create_filter_via_percentile(name="percentile_filter", q1=25, q2=50, q3=75):
# #     def __filter(inputs):
# #         Q1, median, Q3 = tf.split(tfp.stats.percentile(inputs, q=[q1, q2, q3]), 3) # requires tf>=2.8.0
# #         IQR = Q3 - Q1
# #         LO = Q1 - (1.5 * IQR)
# #         HI = Q3 + (1.5 * IQR)
# #         kept = tf.clip_by_value(inputs, LO, HI)
# #         return kept
# #     __filter.__name__ = name
# #     return __filter