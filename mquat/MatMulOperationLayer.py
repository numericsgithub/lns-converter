# -*- coding: utf-8 -*-

import tensorflow as tf
from .OperationLayer import OperationLayer
from .QuantizerBase import NON_QUANT
from .QuantizerBase import DEFAULT_DATATYPE


class MatMulOperationLayer(OperationLayer):
    """matrix multiplication operation layer

    the matrix multiplication is performed in two stages, first the multiplication
    is executed, the results are passed through the multiplication quantisizer,
    then results are summed up, while applying the summation quantisizer after
    each addtion.

    Parameters:
        name (string):
            the name of the layer in the Tensorflow graph.
        dtype (tf.dtypes.DType):
            datatype of the layer's operations.
            default is float32.

    Attributes:
        quant_mul (Quantizer):
            quantisizer that is applied after each multiplication.
        quant_sum (Quantizer):
            quantisizer that is applied after each addition.
    """

    def __init__(self, name, dtype=DEFAULT_DATATYPE):
        super().__init__(name, "matmul", dtype=dtype)
        self.quant_mul = NON_QUANT
        self.quant_sum = NON_QUANT
        self.log_func = lambda x: x

    def isQuantisizing(self):
        """test if the layer performs some quantisized operations

        Returns:
            (bool):
                true if an operation is quantisized.
        """
        return self.quant_mul.isQuantisizing() or self.quant_sum.isQuantisizing() or self.quant_out.isQuantisizing()

    @tf.function
    def call(self, inputs):
        """forward propagation

        apply the layer operation in the forward pass.

        Parameters:
            inputs (list):
                list of the two input tensors.
        Returns:
            (tensor):
                output of the layer.
        """
        self.log_func(inputs)
        x, y = inputs
        # performance optimisation: if no quantisation is applied,
        # use the TensorFlow matrix multiplication operation
        if not self.quant_mul.isQuantisizing() and not self.quant_sum.isQuantisizing():
            return tf.matmul(x, y)

        # apply the matrix multiplication
        # first apply all multiplications that occure during a matrix multiplication
        tmp = tf.einsum('ij,jk->ijk', x, y)
        # quantisize the multiplication result
        tmp_q = self.quant_mul(tmp)
        # performance optimisation: if no summation quantisation is applied,
        # use the TensorFlow summation operation
        if not self.quant_sum.isQuantisizing():
            return tf.reduce_sum(tmp_q, 1)

        # sum up allong the second dimension and apply the sum_quant after each
        # addition
        tmp_sum = tmp[:, 0, :]
        for index_1 in tf.range(1, tmp.shape[1]):
            tmp_sum = self.quant_sum(tmp_sum + tmp[:, index_1, :])

        return tmp_sum

    # y_splits = tf.split(y, 8, axis=1)
    # y_splits_results = []
    # for y_s in y_splits:
    #     # apply the matrix multiplication
    #     # first apply all multiplications that occure during a matrix multiplication
    #     tf.print("SHAPE OF X", tf.shape(x))
    #     tf.print("SHAPE OF y_s", tf.shape(y_s))
    #     tmp = tf.einsum('ij,jk->ijk', x, y_s)
    #     tf.print("SHAPE OF ij,jk->ijk", tf.shape(tmp))
    #     tf.print("SHAPE OF sum(ij,jk->ijk)", tf.shape(tf.reduce_sum(tmp, 1)))
    #     # quantisize the multiplication result
    #     tmp_q = self.quant_mul(tmp)
    #
    #     y_splits_results.append(tf.reduce_sum(tmp_q, 1))
    #     # # performance optimisation: if no summation quantisation is applied,
    #     # # use the TensorFlow summation operation
    #     # if not self.quant_sum.isQuantisizing():
    #     #     return tf.reduce_sum(tmp_q, 1)
    #
    # # # sum up allong the second dimension and apply the sum_quant after each
    # # # addition
    # # tmp_sum = tmp[:, 0, :]
    # # for index_1 in tf.range(1, tmp.shape[1]):
    # #     tmp_sum = self.quant_sum(tmp_sum + tmp[:, index_1, :])