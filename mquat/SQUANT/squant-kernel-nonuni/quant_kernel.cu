// *
// @file Different utility functions
// Copyright (c) Cong Guo, Yuxian Qiu, Jingwen Leng, Xiaotian Gao,
// Chen Zhang, Yunxin Liu, Fan Yang, Yuhao Zhu, Minyi Guo
// All rights reserved.
// This file is part of SQuant repository.
//
// SQuant is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// SQuant is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with SQuant repository.  If not, see <http://www.gnu.org/licenses/>.
// *

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <assert.h>
#include <stdio.h>
using namespace std;
namespace {

template <typename scalar_t>
__global__ void quant_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> x, // Already quantized weights (maybe float or int dont know) numbers between -128 and 127
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> y, // 1D INT Array [-128, -127, -126, ..., -1, 0, 1, ..., 127] SORTED!!!
    size_t x_size,
    size_t y_size, // 256   because there are 256 different possible numbers
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> z,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> tensor_idx) // DONT CARE! never used
{
    __shared__ float y_shared[256];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x < y_size) y_shared[threadIdx.x] = y[threadIdx.x]; // Just a cached y. Nothing more!!
    __syncthreads();
    float sub_min = 102400.0;
    float z_min = 0.0;
    float idx_min = 0.0;
    if(idx < x_size) {
        float x_v = x[idx]; // pre quantized value (Was pre quantized in quant_affine.py:linear_quantize(...,inplace=False)
        for(int i = 0; i < y_size; i++){ // this is just a joke! Really! Overcomplicated normal fixed point quantization
            float sub_v = fabsf(x_v - y_shared[i]);
            if(sub_v <= sub_min)
            {
                sub_min = sub_v;
                z_min = y_shared[i];
                idx_min = i;
            }
            else
                break;
        }
        z[idx] = z_min;
        tensor_idx[idx] = idx_min;
    }
}

template <typename scalar_t>
__device__ __forceinline__ void rounding_forward_cuda_kernel(
    scalar_t delta, // IGNORED
    scalar_t rounding_error_sum, // Sum of all errors per channel! after normal fixed quant
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> rounding_number_, // normal fixed rounded weights
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> rounding_error_,  // error of the normal rounding weights thingi

    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> number_, // The number to round to
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> error_,  // The error that occours on that round
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> priority_, // A zero for every weight
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> order_,    // A zero for every weight.

    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> error_1, // IGNORED
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> priority_1, // A zero for every weight but for the opposite round

    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> subset,
    torch::TensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> best_subset,

    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> flip_number_, // IGNORED
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> flip_number, // IGNORED

    const size_t en,
    const size_t oc,
    const size_t ic
)
{
    int last_changed = 0;
    rounding_error_sum = fabsf(rounding_error_sum);
    auto idx = order_;
    scalar_t topk = fabsf(rounding_error_sum);
    bool over_calibration = (topk >= fabsf(rounding_error_sum)); // Only true when topk was ROUNDED UP

    scalar_t sum = 0;
    int best_subset_size = rounding_error_.size(0);
    int cur_size = 0;
    size_t n = rounding_error_.size(0);
    size_t max_n = 12;
    if(max_n > n)
    {
        max_n = n;
    }

    scalar_t new_topk = topk ; // new topk is the rest of the top k that was not reached yet
    scalar_t closestSum = new_topk;
    size_t new_last_changed = 0;
    bool last_round = false;
    while (last_changed < rounding_error_.size(0))
    {
        if(n > rounding_error_.size(0) - last_changed)
        {
            n = rounding_error_.size(0) - last_changed;
        }
        if(n > max_n)
        {
            n = max_n;
        }
        if(n > 0)
        {
            for (int i = 0; i < rounding_error_.size(0); i++)
            {
                size_t idx_ = idx[i];
                subset[idx_] = 0.0;
            }
            for (int i = 0; i < (1 << n); i++)
            {
                sum = 0.0;
                cur_size = 0;
                for (int j = 0; j < n; j++)
                {
                    size_t idx_ = idx[j];
                    size_t idx__ = idx[j + last_changed];
                    if ((i & (1 << j)) != 0 && fabsf(number_[idx__] - rounding_number_[idx__]) != 0)
                    {
                        cur_size++;
                        sum += fabsf(number_[idx__] - rounding_number_[idx__]);
                        new_last_changed = j;
                        subset[idx_] = 1.0;
                    }
                    else
                    {
                        subset[idx_] = 0.0;
                    }
                }

                // Check if the current sum is closer to the target than the current closest sum
                bool cond = (new_topk - sum < closestSum && new_topk - sum > 0.0) || (new_topk - sum == closestSum  && cur_size < best_subset_size);
                if (last_round)
                    cond = fabsf(new_topk - sum) < closestSum || (fabsf(new_topk - sum) == closestSum  && cur_size < best_subset_size);
                if (cond)
                {
                    best_subset_size = cur_size;
                    closestSum = fabsf(new_topk - sum);
                    for (int k = 0; k < n; k++)
                    {
                        size_t idx_ = idx[k];
                        best_subset[idx_] = subset[idx_];
                    }
                }
            }

            // Apply best set
            int best_subset_last_true_index = 0;
            for (int i = 0; i < n; i++)
            {
                size_t idx_ = idx[i];
                if(best_subset[idx_] == 1.0)
                {
                    size_t idx__ = idx[i + last_changed];
                    rounding_error_[idx__] =  error_[idx__];
                    rounding_number_[idx__] = number_[idx__];
                    best_subset_last_true_index = i;
                }
            }

            new_topk = closestSum;
            if(new_last_changed == 0)
            {
                if(last_round)
                    break;
                else
                {
                    last_round = true;
                    max_n = 16;
                }
            }
            else
            {
                last_changed += new_last_changed + 1;
            }
        }
        else
        {
            break;
        }
    }

    // If took roundings are too much -> Prioritize the last rounding. Maybe to be rounded in the other direction again later??
    // This priority stuff is used! After Rounding by kernel groups the channel groups use this! quant_modules.py:220
    if(closestSum > topk)
    {
        size_t idx_c = idx[last_changed  - 1];
        priority_1[idx_c] = fabsf(rounding_error_[idx_c]);
    }
    else
    {
        if(last_changed < rounding_error_.size(0))
        {
            size_t idx_c = idx[last_changed ];
            priority_[idx_c] = fabsf(rounding_error_[idx_c]);
        }
    }
}

template <typename scalar_t>
__global__ void rounding_loop_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> flip_number,  // IGNORED Scalar = 0.0
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> flip_up_number,  // IGNORED Scalar = 0.0
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> flip_down_number,  // IGNORED Scalar = 0.0

    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> rounding_error_sum, // Sum of all errors per channel! after normal fixed quant
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> rounding_number_, // normal fixed rounded weights
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> rounding_error_, // error of the normal rounding weights thingi

    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> up_number_, // the number when you change rounding to UP
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> up_error_,  // the error when you change rounding to UP
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> up_priority_, // All zero for each weight
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> up_order_, // up_priority sort by size biggest errors first BUT ONLY THE INDECIES

    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> down_number_, // the number when you change rounding to DOWN
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> down_error_, // the error when you change rounding to DOWN
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> down_priority_, // All zero for each weight
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> down_order_, // down_priority sort by size biggest errors first BUT ONLY THE INDECIES

    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> subset,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> best_subset,

    const size_t input_channel,
    const size_t element_number
)
{
    const int oc = blockIdx.y;
    const int ic = blockIdx.x * blockDim.x + threadIdx.x;
    if (ic >= input_channel) return;

    if(rounding_error_sum[oc][ic] < 0)
    {
        // UP
        scalar_t delta = 1.0;
        rounding_forward_cuda_kernel(
            delta, // IGNORED
            rounding_error_sum[oc][ic],
            rounding_number_[oc][ic],
            rounding_error_[oc][ic],

            up_number_[oc][ic],
            up_error_[oc][ic],
            up_priority_[oc][ic],
            up_order_[oc][ic],

            down_error_[oc][ic], // IGNORED
            down_priority_[oc][ic],

            subset[oc][ic],
            best_subset[oc][ic],

            flip_up_number,  // IGNORED
            flip_number,  // IGNORED

            element_number,
            oc,
            ic
        );
    }
    else
    {
        // Down
        scalar_t delta = -1.0;
        rounding_forward_cuda_kernel(
            delta, // IGNORED
            rounding_error_sum[oc][ic],
            rounding_number_[oc][ic],
            rounding_error_[oc][ic],

            down_number_[oc][ic],
            down_error_[oc][ic],
            down_priority_[oc][ic],
            down_order_[oc][ic],

            up_error_[oc][ic], // IGNORED
            up_priority_[oc][ic],

            subset[oc][ic],
            best_subset[oc][ic],

            flip_down_number,  // IGNORED
            flip_number,  // IGNORED

            element_number,
            oc,
            ic
        );
    }

    return;
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor>  quant_forward_cuda(
    torch::Tensor x, // Already a bit quantized weights (floats) numbers between -128 and 127
    torch::Tensor y) // 1D INT Array [-128, -127, -126, ..., -1, 0, 1, ..., 127]
{
    const int threads = 1024;
    const dim3 blocks((x.size(0) + threads - 1) / threads);
    auto z   = torch::zeros_like(x);
    auto idx = torch::zeros_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "quant_forward_cuda", ([&] {
        quant_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            x.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            y.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            x.size(0),
            y.size(0), // 256   because there are 256 different possible numbers
            z.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(), // rounded numbers
            idx.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));

    return std::make_tuple(z,idx);
}



void rounding_loop_forward_cuda(
    torch::Tensor flip_number,
    torch::Tensor flip_up_number,
    torch::Tensor flip_down_number,

    torch::Tensor rounding_error_sum,
    torch::Tensor rounding_number_,
    torch::Tensor rounding_error_,

    torch::Tensor up_number_,
    torch::Tensor up_error_,
    torch::Tensor up_priority_,
    torch::Tensor up_order_,

    torch::Tensor down_number_,
    torch::Tensor down_error_,
    torch::Tensor down_priority_,
    torch::Tensor down_order_,

    torch::Tensor subset,
    torch::Tensor best_subset
)
{
    // const dim3 blocks((x.size(0) + threads - 1) / threads);

    const size_t size_0 = rounding_number_.size(0);
    const size_t size_1 = rounding_number_.size(1);
    const size_t size_2 = rounding_number_.size(2);

    const size_t threads = 1;
    const dim3 grid((size_1 + threads - 1) / threads, size_0);

    AT_DISPATCH_FLOATING_TYPES(
        flip_number.type(),
        "rounding_loop_forward_cuda",
        (
            [&] {
                rounding_loop_forward_cuda_kernel<scalar_t><<<grid, threads>>>(
                    flip_number.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
                    flip_up_number.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
                    flip_down_number.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),

                    rounding_error_sum.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    rounding_number_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    rounding_error_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),

                    up_number_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    up_error_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    up_priority_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    up_order_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),

                    down_number_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    down_error_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    down_priority_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    down_order_.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),

                    subset.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                    best_subset.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),

                    size_1,
                    size_2
                );
            }
        )
    );
    return;
}