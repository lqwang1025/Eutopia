/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * (C) COPYRIGHT Daniel Wang Limited.
 * File       : im2col.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-26:16:49:20
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <vector>

#include "op/cpu/compute/im2col.h"
#include "core/logging.h"
#include "core/ir/tensor.h"

namespace eutopia {
namespace op {
namespace cpu {

template<typename T>
T _get_pixel(const core::ir::Tensor* im, int h, int w, int c, const std::vector<int32_t>& pads) {
    std::vector<uint32_t> im_dims = im->dims();
    int width = (int)im_dims[3];
    int height = (int)im_dims[2];
    int channel = (int)im_dims[1];
    h -= (pads[0]+pads[1]);
    w -= (pads[2]+pads[3]);
    if (h < 0 ||
        w < 0 ||
        h >= height ||
        w >= width) return (T)0;
    return im->data<T>(w + width*(h + height*c));
}

void im2col(const Convolution2DParam* op_param, const core::ir::Tensor* im, core::ir::Tensor& col) {
    std::vector<uint32_t> kernel_shape = op_param->kernel_shape;
    std::vector<uint32_t> strides      = op_param->strides;
    std::vector<int32_t> pads          = op_param->pads; // t l b r
    CHECK(pads.size() == 4,);
    CHECK(kernel_shape.size() == 4,);
    CHECK(strides.size() == 2,);
    std::vector<uint32_t> im_dims = im->dims();
    CHECK(im_dims.size() == 4,);
    std::vector<uint32_t> col_dims = col.dims();
    CHECK(col_dims.size() == 2,);
    uint32_t kh = kernel_shape[2];
    uint32_t kw = kernel_shape[3];
    uint32_t channels = im_dims[1];
    int channels_col = (int)channels * kh * kw;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kw;
        int h_offset = (c / kw) % kh;
        int c_im = c / kw / kh;
        for (int h = 0; h < (int)col_dims[0]; ++h) {
            for (int w = 0; w < (int)col_dims[1]; ++w) {
                int im_h = h_offset + h * strides[0];
                int im_w = w_offset + w * strides[1];
                int col_index = (c * (int)col_dims[0] + h) * (int)col_dims[1] + w;
                col.mutable_data<float>((uint32_t)col_index) = _get_pixel<float>(im, im_h, im_w, c_im, pads);
            }
        }
    }
}

void col2im(const Convolution2DParam* op_param,const core::ir::Tensor* col, core::ir::Tensor& im) {
    
}

template float _get_pixel<float>(const core::ir::Tensor* im, int h, int w, int channel, const std::vector<int32_t>& pads);
template uint8_t _get_pixel<uint8_t>(const core::ir::Tensor* im, int h, int w, int channel, const std::vector<int32_t>& pads);
template int8_t _get_pixel<int8_t>(const core::ir::Tensor* im, int h, int w, int channel, const std::vector<int32_t>& pads);

} // namespace cpu
} // namespace op
} // namespace eutopia
