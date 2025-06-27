# *
# @file Different utility functions
# Copyright (c) Cong Guo, Yuxian Qiu, Jingwen Leng, Xiaotian Gao,
# Chen Zhang, Yunxin Liu, Fan Yang, Yuhao Zhu, Minyi Guo
# All rights reserved.
# This file is part of SQuant repository.
#
# SQuant is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SQuant is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SQuant repository.  If not, see <http://www.gnu.org/licenses/>.
# *
import logging
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import scipy.stats as st
# from squant.squant_tests import x_quant

IS_INIT_FINISHED = False
# import MetaParameters as mp
# from quant_affine import *

# def _quantization(tensor, quant_value):
#     shape = tensor.shape
#     quant_tensor = tensor.view(-1)
#     quant_value = quant_value.type_as(quant_tensor)
#     quant_tensor, quant_idx = quant_cuda.quant(quant_tensor, quant_value)
#     quant_tensor = quant_tensor.view(shape)
#     quant_idx = quant_idx.view(shape).type(torch.long)
#     return quant_tensor, quant_idx

def logstuff(prefix_str, nptab):
    pass
    # nptab = nptab.cpu().clone().numpy()
    # min = np.min(nptab)
    # max = np.max(nptab)
    # variance = np.var(nptab)
    # print(prefix_str, "[{:.2f}; {:.2f}]".format(min, max), "var={:.2f}".format(variance))


def main_logstuff2(prefix_str, coeffs, x, x_quant, x_quant_old, up_priority, down_priority, up_change, down_change, up_number, down_number):
    return
    coeffs = coeffs.cpu().clone().numpy()
    x = x.cpu().clone().numpy()
    x_quant = x_quant.cpu().clone().numpy()
    x_quant_old = x_quant_old.cpu().clone().numpy()
    up_priority = up_priority.cpu().clone().numpy()
    down_priority = down_priority.cpu().clone().numpy()
    up_change = up_change.cpu().clone().numpy()
    down_change = down_change.cpu().clone().numpy()
    up_number = up_number.cpu().clone().numpy()
    down_number = down_number.cpu().clone().numpy()

    squant_changes_mask = x_quant_old == x_quant
    # logstuff2(prefix_str+"x ", squant_changes_mask, x)
    logstuff2(prefix_str+"x_quant ", squant_changes_mask, x_quant_old)
    logstuff2(prefix_str+"up_number ", squant_changes_mask, up_number)
    logstuff2(prefix_str+"down_number ", squant_changes_mask, down_number)
    # logstuff2(prefix_str+"up_priority ", squant_changes_mask, up_priority)
    # logstuff2(prefix_str+"down_priority ", squant_changes_mask, down_priority)

    rounded_up_mask = x_quant > x_quant_old
    rounded_down_mask = x_quant < x_quant_old

    fig, (ax1, ax2) = plt.subplots(2, 1)
    all_x_ticks = []
    all_errors_rounding_up = []
    all_errors_rounding_down = []
    all_cases_rounding_up = []
    all_cases_rounding_down = []
    all_cases_no_rounding = []

    for i, coeff in enumerate(coeffs):
        # cases where squant rounded up for the current coefficient
        mask = rounded_up_mask & (x_quant_old == coeff)
        # We now look at how much the original values deviate from the squanted
        error_rounding_up = x_quant[mask] - x_quant_old[mask]

        # cases where squant rounded down for the current coefficient
        mask = rounded_down_mask & (x_quant_old == coeff)
        # We now look at how much the original values deviate from the squanted
        error_rounding_down = x_quant[mask] - x_quant_old[mask]
        if np.size(error_rounding_up) + np.size(error_rounding_down) > 0:
            all_errors_rounding_up.append(np.abs(np.sum(error_rounding_up)))
            all_errors_rounding_down.append(np.abs(np.sum(error_rounding_down)))
            all_cases_rounding_up.append(np.size(error_rounding_up))
            all_cases_rounding_down.append(np.size(error_rounding_down))
            all_x_ticks.append(coeff)
        all_cases_no_rounding.append(np.size(x_quant_old[x_quant_old == coeff]) - np.size(error_rounding_up) + np.size(error_rounding_down))

    if len(all_x_ticks) == 0:
        all_x_ticks.append(0)

    bar_width = np.max(np.abs([np.min(all_x_ticks)*1.05, np.max(all_x_ticks)*1.05])) / 640.0
    bar_offset = bar_width/2.0

    all_x_ticks = np.array(all_x_ticks)

    ax1.set_xlim((np.min(all_x_ticks)*1.05, np.max(all_x_ticks)*1.05))

    x_min, x_max = np.min(x), np.max(x)
    x_steps = (x_max - x_min)/ (2048.0 * 2.0)
    y1, x1 = np.histogram(np.reshape(x, -1), bins=np.arange(x_min, x_max, x_steps))
    y1 = y1 / np.max(y1)
    y1 = y1 * np.maximum(np.max(all_errors_rounding_up), np.max(all_errors_rounding_down))
    #ax1.hist(np.reshape(x, -1), density=True, bins=2048, label="Data")
    ax1.bar(x1[:-1], y1, width=x_steps, color="lightgrey")

    # mn, mx = (np.min(all_x_ticks)*1.05, np.max(all_x_ticks)*1.05)
    # kde_xs = np.linspace(mn, mx, 2048)
    # kde = st.gaussian_kde(np.reshape(x, -1))
    # ax1.bar(kde_xs, kde.pdf(kde_xs) , label="PDF")
    # print("(np.min(all_x_ticks)*1.05, np.max(all_x_ticks)*1.05)", (np.min(all_x_ticks)*1.05, np.max(all_x_ticks)*1.05))

    ax1.bar(all_x_ticks + bar_offset, all_errors_rounding_up, color='green', width=bar_width, label="round up")
    ax1.bar(all_x_ticks - bar_offset, all_errors_rounding_down, color='crimson', width=bar_width, label="round down")
    ax1.set_xticks(coeffs, rotation=45)
    ax1.set_xticklabels(coeffs, rotation=45)
    ax1.set_xlim((np.min(all_x_ticks)*1.05, np.max(all_x_ticks)*1.05))
    for index, label in enumerate(ax1.xaxis.get_ticklabels()):
        if coeffs[index] not in all_x_ticks:
            label.set_visible(False)
    ax1.set_ylabel("abs(sum())")



    ax2.bar(all_x_ticks + bar_offset, all_cases_rounding_up, color='green', width=bar_width, label="round up")
    ax2.bar(all_x_ticks - bar_offset, all_cases_rounding_down, color='red', width=bar_width, label="round down")
    # ax2.bar(coeffs, all_cases_no_rounding, color='black', width=bar_width, label="no rounding", alpha=0.3)
    ax2.set_xticks(coeffs, rotation=45)
    ax2.set_xticklabels(coeffs, rotation=45)
    ax2.set_xlim((np.min(all_x_ticks)*1.05, np.max(all_x_ticks)*1.05))
    for index, label in enumerate(ax2.xaxis.get_ticklabels()):
        if coeffs[index] not in all_x_ticks:
            label.set_visible(False)
    ax2.set_ylabel("count()")

    rounding_error_sum_before = np.sum(x - x_quant_old)
    rounding_error_sum_after = np.sum(x - x_quant)

    fig.suptitle(prefix_str + " {:.4f} -> {:.4f}".format(rounding_error_sum_before, rounding_error_sum_after) )
    plt.subplots_adjust(left=0.05, bottom=0.11, right=0.995, top=0.95, wspace=0.205, hspace=0.29)
    plt.show()

    # bar_width = 0.05
    # bar_offset = bar_width/2.0
    # all_uniques, all_counts, uniques, counts = logstuff2(prefix_str+"(up_number - x_quant) ", squant_changes_mask, up_change)
    # selection = up_change[squant_changes_mask]
    # all_x_ticks = []
    # for i, _ in enumerate(all_counts):
    #     for j, _ in enumerate(counts):
    #         if uniques[j] == all_uniques[i]:
    #             all_counts[i] -= counts[j]
    #     if all_counts[i] > 0:
    #         if all_uniques[i] not in all_x_ticks:
    #             all_x_ticks.append(all_uniques[i])
    # ax1.bar(all_uniques + bar_offset, all_counts, color='green', width=bar_width, label="rounded up")
    #
    # all_uniques, all_counts, uniques, counts = logstuff2(prefix_str+"(down_number - x_quant) ", squant_changes_mask, down_change)
    # for i, val_i in enumerate(all_counts):
    #     for j, val_j in enumerate(counts):
    #         if uniques[j] == all_uniques[i]:
    #             all_counts[i] -= counts[j]
    #     if all_counts[i] > 0:
    #         if all_uniques[i] not in all_x_ticks:
    #             all_x_ticks.append(all_uniques[i])
    # ax1.bar(all_uniques - bar_offset, all_counts, color='red', width=bar_width, label="rounded down")
    # ax1.set_xticks(all_x_ticks, rotation=45)
    # ax1.set_xticklabels(all_x_ticks, rotation=45)
    # ax1.set_xlim((np.min(all_x_ticks)*1.05, np.max(all_x_ticks)*1.05))
    #
    #
    #
    # ax2.set_xticks(all_x_ticks, rotation=45)
    # ax2.set_xticklabels(all_x_ticks, rotation=45)
    # ax2.set_xlim((np.min(all_x_ticks)*1.05, np.max(all_x_ticks)*1.05))
    # plt.show()




def logstuff2(prefix_str, squant_changes_mask, nptab):
    selection = nptab[squant_changes_mask]
    min = np.min(selection)
    max = np.max(selection)
    variance = np.var(selection)
    uniques, counts = np.unique(selection, return_counts=True)
    all_uniques, all_counts = np.unique(nptab, return_counts=True)
    print(prefix_str, "[{:.2f}; {:.2f}]".format(min, max), "uniques:", uniques.tolist(), counts.tolist())
    return all_uniques, all_counts, uniques, counts

def adaptive_round(var_name, x, x_quant, coeffs, isNonUniform, squant_c=True, squant_k=True):
    """
    Applies squant on the x_quant weights with regards to the full-precision weights x.
    If not isNonUniform -> The original squant is used. Otherwise, the new implementation is used for
    non uniform quantization.
    Args:
        x: Full-precision weights
        x_quant: The quantized full-precision weights.
        coeffs: All coefficients allowed by the applied quantization.
        isNonUniform: Determines which squant implementation should be used.
        squant_c: apply squant to each channel
        squant_k: apply squant to each kernel (if present).

    Returns: the quantized weights with squant applied to them.

    """
    # print("Start import of torch and torch cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import quant_cuda as squant
    import quant_cuda_nonuni as squant_nonuni
    # print("Import of torch and torch cuda is now done")
    if isNonUniform:
        quant_cuda = squant_nonuni
    else:
        quant_cuda = squant
    x = torch.tensor(x).cuda()
    x_quant = torch.tensor(x_quant).cuda()
    coeffs = torch.tensor(coeffs).cuda()

    def f(list):
        text = ""
        for l in list:
            text += "{:.2f}".format(l) + "\t"#.rjust(7)
        return text

    rounding_error = x_quant - x  # Like 2 - 1.8 = 0.2
    logstuff("rounding_error ", rounding_error)

    # print("rounding_error", rounding_error)
    subsets = rounding_error.clone()
    best_subset = rounding_error.clone()

    t_max = torch.max(coeffs)
    t_min = torch.min(coeffs)

    def custom_up_down(allowed_values, inc_or_dec, inc_dec_number, inc_dec_error):
        # print("inc_dec_number", inc_dec_number)
        # print("inc_dec_error", inc_dec_error)
        for index, allowed_val in enumerate(allowed_values):
            # print(allowed_val, x_quant[x_quant == allowed_val])
            if index + inc_or_dec >= len(allowed_values):
                continue
            if index + inc_or_dec < 0:
                continue
            new_val = allowed_values[index + inc_or_dec]
            new_val = new_val.type_as(inc_dec_number)
            mask = torch.logical_and(x_quant == allowed_val, inc_dec_error != 0.0)
            inc_dec_error[mask] += new_val - inc_dec_number[mask]
            inc_dec_number[mask] = new_val
        # print("inc_dec_number", inc_dec_number)
        # print("inc_dec_error", inc_dec_error)

    # This stuff describes for each rounded weight:
    # What error and number it will be when you round the other way around...
    # Here for the already DOWN ROUNDED weights the error and number for up rounding
    up_number = x_quant.clone()  # the normal quantized weights
    up_error = rounding_error.clone()  # the normal quantized weights - weights
    up_error[x >= t_max] = 0.0
    up_error[up_error > 0] = 0.0  # Only keep those errors where the weight was ROUNDED DOWN
    up_priority = up_error.clone().abs()  # All errors for rounding up as absolut values

    coeffs, _ = torch.sort(coeffs, descending=False)
    custom_up_down(coeffs, 1, up_number, up_error)

    # print("x", f(np.reshape(x.clone().cpu().numpy(), -1).tolist()))
    # print("x_quant", f(np.reshape(x_quant.clone().cpu().numpy(), -1).tolist()))
    # print("rounding_error", f(np.reshape(rounding_error.clone().cpu().numpy(), -1).tolist()))
    #
    # print("up_number", f(np.reshape(up_number.clone().cpu().numpy(), -1).tolist()))
    # print("up_error", f(np.reshape(up_error.clone().cpu().numpy(), -1).tolist()))

    # up_error[up_error != 0]  += 1 # +1 those errors where the weight was ROUNDED DOWN
    # up_number[up_error != 0] += 1 # +1 those quantized weights where the weight was ROUNDED DOWN

    # Here for the already UP ROUNDED weights the error and number for down rounding
    down_number = x_quant.clone()
    down_error = rounding_error.clone()
    down_error[x <= t_min] = 0.0
    down_error[down_error < 0] = 0.0
    down_priority = down_error.clone().abs()  # All errors for rounding down as absolut values

    custom_up_down(coeffs, -1, down_number, down_error)
    # print("down_number", f(np.reshape(down_number.clone().cpu().numpy(), -1).tolist()))
    # print("down_error", f(np.reshape(down_error.clone().cpu().numpy(), -1).tolist()))
    # down_error[down_error != 0]  -= 1
    # down_number[down_error != 0] -= 1

    flip_number = torch.tensor([0.0], device=x.device)
    flip_up_number = torch.tensor([0.0], device=x.device)
    flip_down_number = torch.tensor([0.0], device=x.device)

    # print("x.shape", x.shape)
    conver_shape = x.view(x.shape[0], x.shape[1], -1).shape
    if conver_shape[2] == 1:
        squant_k = False

    results = rounding_error.view(conver_shape).sum(-1).cpu().clone().numpy()
    results_abs_sum = np.sum(np.abs(results))
    results_sum = np.abs(np.sum(results))

    if squant_k:
        print("squant_k")
        logstuff("x ", x.view(conver_shape))
        logstuff("x_quant ", x_quant.view(conver_shape))
        logstuff("up_priority ", up_priority.view(conver_shape))
        logstuff("down_priority ", down_priority.view(conver_shape))
        logstuff("down_number HERE!", down_number.view(conver_shape))
        logstuff("up_number ", up_number.view(conver_shape))
        logstuff("(up_number - x_quant) ", (up_number - x_quant).view(conver_shape))
        logstuff("(down_number - x_quant) ", (down_number - x_quant).view(conver_shape))
        x_quant_copy = x_quant.clone()
        up_number_copy = up_number.clone()
        down_number_copy = down_number.clone()

        # results = rounding_error.view(conver_shape).sum(-1).cpu().clone().numpy()
        # results_abs_sum = np.sum(np.abs(results))
        # results_sum = np.abs(np.sum(results))
        # print("Before Squant K rounding_error_sum", results_abs_sum, results_sum)
        rounding_error_sum = rounding_error.view(conver_shape).sum(-1)
        # up_priority = (up_number - x_quant) + up_priority
        # down_priority = (down_number - x_quant) + down_priority
        _, up_order = torch.sort(up_priority.view(conver_shape), descending=True)  # up_priority sort by size biggest errors first BUT ONLY THE INDECIES
        _, down_order = torch.sort(down_priority.view(conver_shape), descending=True)  # down_priority sort by size biggest errors first BUT ONLY THE INDECIES
        # _, up_order = torch.sort((up_number - x_quant).view(conver_shape), descending=True)
        # _, down_order = torch.sort((down_number - x_quant).view(conver_shape), descending=True)
        up_priority *= 0.0  #
        down_priority *= 0.0  #
        subsets *= 0.0
        best_subset *= 0.0
        if not isNonUniform:
            quant_cuda.rounding_loop(
                flip_number,  # IGNORED
                flip_up_number,  # IGNORED
                flip_down_number,  # IGNORED

                rounding_error_sum,  # all rounding errors summed up
                x_quant.view(conver_shape),
                rounding_error.view(conver_shape),

                up_number.view(conver_shape),
                up_error.view(conver_shape),
                up_priority.view(conver_shape),
                up_order.type_as(up_priority).view(conver_shape),

                down_number.view(conver_shape),
                down_error.view(conver_shape),
                down_priority.view(conver_shape),
                down_order.type_as(down_priority).view(conver_shape),
            )
        else:
            quant_cuda.rounding_loop(
                flip_number,  # IGNORED
                flip_up_number,  # IGNORED
                flip_down_number,  # IGNORED

                rounding_error_sum,  # all rounding errors summed up
                x_quant.view(conver_shape),
                rounding_error.view(conver_shape),

                up_number.view(conver_shape),
                up_error.view(conver_shape),
                up_priority.view(conver_shape),
                up_order.type_as(up_priority).view(conver_shape),

                down_number.view(conver_shape),
                down_error.view(conver_shape),
                down_priority.view(conver_shape),
                down_order.type_as(down_priority).view(conver_shape),

                subsets.view(conver_shape),
                best_subset.view(conver_shape),
            )

        main_logstuff2(var_name + " SQUANT_K", coeffs, x.view(conver_shape), x_quant.view(conver_shape), x_quant_copy.view(conver_shape),
                       up_priority.view(conver_shape), down_priority.view(conver_shape),
                       (up_number_copy - x_quant_copy).view(conver_shape), (down_number_copy - x_quant_copy).view(conver_shape),
                       up_number.view(conver_shape), down_number.view(conver_shape))
        # logstuff2("AFTER SQUANT K: x ", squant_changes_mask, x.view(conver_shape))
        # logstuff2("AFTER SQUANT K: x_quant ", squant_changes_mask, x_quant_copy.view(conver_shape))
        # # logstuff2("AFTER SQUANT K: up_priority ", squant_changes_mask, up_priority.view(conver_shape))
        # # logstuff2("AFTER SQUANT K: down_priority ", squant_changes_mask, down_priority.view(conver_shape))
        # logstuff2("AFTER SQUANT K: (up_number - x_quant) ", squant_changes_mask, (up_number_copy - x_quant_copy).view(conver_shape))
        # logstuff2("AFTER SQUANT K: (down_number - x_quant) ", squant_changes_mask, (down_number_copy - x_quant_copy).view(conver_shape))


        # results = rounding_error.view(conver_shape).sum(-1)
        # results_abs_sum = np.sum(np.abs(results))
        # results_sum = np.abs(np.sum(results))
        # print("After Squant K rounding_error_sum", results_abs_sum, results_sum)

    # up_priority_extra = (up_number - x_quant)
    # up_priority_extra[x <= 0] = 0.0
    # up_priority = up_priority_extra + up_priority
    #
    #
    # down_priority_extra = (down_number - x_quant)
    # down_priority_extra[x >= 0] = 0.0
    # down_priority = down_priority_extra + down_priority

    # up_priority = (up_number - x_quant) + up_priority
    # down_priority = (down_number - x_quant) + down_priority

    if squant_c:
        print("squant_c")
        conver_shape = (1, x.shape[0], -1)
        # logstuff("x ", x.view(conver_shape))
        # logstuff("x_quant ", x_quant.view(conver_shape))
        # logstuff("up_priority ", up_priority.view(conver_shape))
        # logstuff("down_priority ", down_priority.view(conver_shape))
        # logstuff("(up_number - x_quant) ", (up_number - x_quant).view(conver_shape))
        # logstuff("(up_number - x_quant) ", (up_number - x_quant).view(conver_shape))
        x_quant_copy = x_quant.view(conver_shape).clone()
        up_number_copy = up_number.view(conver_shape).clone()
        down_number_copy = down_number.view(conver_shape).clone()
        # results = rounding_error.view(conver_shape).sum(-1).cpu().clone().numpy()
        # results_abs_sum = np.sum(np.abs(results))
        # results_sum = np.abs(np.sum(results))
        # print("Before Squant C rounding_error_sum", results_abs_sum, results_sum)
        rounding_error_sum = rounding_error.view(conver_shape).sum(-1)

        # _, up_order = torch.sort((up_number - x_quant).view(conver_shape), descending=True)
        # _, down_order = torch.sort((down_number - x_quant).view(conver_shape), descending=True)
        # print("LOOK HERE ", np.reshape(up_priority.cpu().clone().numpy(), -1).tolist())
        # print("LOOK HERE ", np.reshape(down_priority.cpu().clone().numpy(), -1).tolist())
        # print("LOOK HERE ", np.reshape(up_number, -1).tolist())
        #print("LOOK HERE ", np.reshape((down_number - x_quant).cpu().clone().numpy(), -1).tolist())
        _, up_order = torch.sort(up_priority.view(conver_shape), descending=True)
        _, down_order = torch.sort(down_priority.view(conver_shape), descending=True)
        subsets *= 0.0
        best_subset *= 0.0
        if not isNonUniform:
            quant_cuda.rounding_loop(
                flip_number,
                flip_up_number,
                flip_down_number,

                rounding_error_sum,
                x_quant.view(conver_shape),
                rounding_error.view(conver_shape),

                up_number.view(conver_shape),
                up_error.view(conver_shape),
                up_priority.view(conver_shape),
                up_order.type_as(up_priority).view(conver_shape),

                down_number.view(conver_shape),
                down_error.view(conver_shape),
                down_priority.view(conver_shape),
                down_order.type_as(down_priority).view(conver_shape)
            )
        else:
            quant_cuda.rounding_loop(
                flip_number,
                flip_up_number,
                flip_down_number,

                rounding_error_sum,
                x_quant.view(conver_shape),
                rounding_error.view(conver_shape),

                up_number.view(conver_shape),
                up_error.view(conver_shape),
                up_priority.view(conver_shape),
                up_order.type_as(up_priority).view(conver_shape),

                down_number.view(conver_shape),
                down_error.view(conver_shape),
                down_priority.view(conver_shape),
                down_order.type_as(down_priority).view(conver_shape),

                subsets.view(conver_shape),
                best_subset.view(conver_shape),
            )
        main_logstuff2(var_name + " SQUANT_C", coeffs, x.view(conver_shape), x_quant.view(conver_shape), x_quant_copy.view(conver_shape),
                       up_priority.view(conver_shape), down_priority.view(conver_shape),
                       (up_number_copy - x_quant_copy).view(conver_shape), (down_number_copy - x_quant_copy).view(conver_shape),
                       up_number.view(conver_shape), down_number.view(conver_shape))
        results = rounding_error.view(conver_shape).sum(-1).cpu().clone().numpy()
        results_abs_sum = np.sum(np.abs(results))
        results_sum = np.abs(np.sum(results))

        # print("Before Squant C rounding_error_sum", results_abs_sum, results_sum)

    # if squant_c:
    #     conver_shape = (1, 1, -1)
    #     results = rounding_error.view(conver_shape).sum(-1).cpu().clone().numpy()
    #     results_abs_sum = np.sum(np.abs(results))
    #     results_sum = np.sum(results)
    #     print("Before Squant C rounding_error_sum", results_abs_sum, results_sum)
    #     rounding_error_sum = rounding_error.view(conver_shape).sum(-1)
    #
    #     _, up_order = torch.sort(up_priority.view(conver_shape), descending=True)
    #     _, down_order = torch.sort(down_priority.view(conver_shape), descending=True)
    #     quant_cuda.rounding_loop(
    #         flip_number,
    #         flip_up_number,
    #         flip_down_number,
    #
    #         rounding_error_sum,
    #         x_quant.view(conver_shape),
    #         rounding_error.view(conver_shape),
    #
    #         up_number.view(conver_shape),
    #         up_error.view(conver_shape),
    #         up_priority.view(conver_shape),
    #         up_order.type_as(up_priority).view(conver_shape),
    #
    #         down_number.view(conver_shape),
    #         down_error.view(conver_shape),
    #         down_priority.view(conver_shape),
    #         down_order.type_as(down_priority).view(conver_shape)
    #     )
    #     results = rounding_error.view(conver_shape).sum(-1).cpu().clone().numpy()
    #     results_abs_sum = np.sum(np.abs(results))
    #     results_sum = np.sum(results)
    #     print("Before Squant C rounding_error_sum", results_abs_sum, results_sum)

    return x_quant.clone().cpu().numpy(), results_abs_sum, results_sum
