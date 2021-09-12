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
 * File       : blas.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-26:16:49:20
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <vector>
#include <omp.h>

#include "core/ir/tensor.h"
#include "core/logging.h"

#include "op/cpu/compute/blas.h"
#include "op/cpu/compute/avtivation.h"

namespace eutopia {
namespace op {
namespace cpu {

void gemm(const core::ir::Tensor* weight, core::ir::Tensor* data_col, core::ir::Tensor* result) {
    std::vector<uint32_t> weight_shape = weight->dims();
    CHECK(weight_shape.size()==4,);
    std::vector<uint32_t> data_shape = data_col->dims();
    CHECK(data_shape.size()==3,);
    uint32_t weight_w = weight_shape[3]*weight_shape[2]*weight_shape[1];
    uint32_t B = data_shape[0];
    uint32_t H = weight_shape[0];
    uint32_t W = data_shape[2];
    uint32_t K = data_shape[1];
    std::vector<uint32_t> res_shape = result->dims();
    CHECK(res_shape.size()==4,);
    int n, h, w, k;
    int data_offset, result_offset;
    int tid;
#pragma omp parallel for private(data_offset, result_offset, n, h, w, k)
    for (n = 0; n < (int)B; ++n) {
        data_offset = n*data_shape[1]*data_shape[2];
        result_offset = n*res_shape[1]*res_shape[2]*res_shape[3];
        for (h = 0; h < (int)H; ++h) {
            for (k = 0; k < (int)K; ++k) {
                register float a_part = weight->data<float>(h*weight_w+k);
                for (w = 0; w < W; ++w) {
                    tid = omp_get_thread_num();
                    result->mutable_data<float>(result_offset+h*W+w) += a_part*data_col->data<float>(data_offset+k*W+w);
                }
            }
        }
    }
}

void add_bias(const core::ir::Tensor* bias, core::ir::Tensor* result, const std::string& act_type) {
    std::vector<uint32_t> res_shape = result->dims(); // n c h w
    ActivationFunc activate = get_activation(act_type);
    CHECK(activate!=nullptr,);
    CHECK(res_shape.size()==4,);
    CHECK(bias->dims()[0]==res_shape[1],);
    for (int  n = 0; n < (int)res_shape[0]; ++n) {
        int n_offset = n*res_shape[1]*res_shape[2]*res_shape[3];
        for (int c = 0; c < (int)res_shape[1]; ++c) {
            int c_offset = c*res_shape[2]*res_shape[3];
            for (int pl = 0; pl < (int)res_shape[2]*res_shape[3]; ++pl) {
                result->mutable_data<float>(n_offset+c_offset+pl) += bias->data<float>(c);
                result->mutable_data<float>(n_offset+c_offset+pl) = activate(result->data<float>(n_offset+c_offset+pl));
            }
        }
    }
}

void scale_result(const core::ir::Tensor* scale, core::ir::Tensor* result) {
    std::vector<uint32_t> res_shape = result->dims(); // n c h w
    CHECK(res_shape.size()==4,);
    CHECK(scale->dims()[0]==res_shape[1],);
    for (int  n = 0; n < (int)res_shape[0]; ++n) {
        int n_offset = n*res_shape[1]*res_shape[2]*res_shape[3];
        for (int c = 0; c < (int)res_shape[1]; ++c) {
            int c_offset = c*res_shape[2]*res_shape[3];
            for (int pl = 0; pl < (int)res_shape[2]*res_shape[3]; ++pl) {
                result->mutable_data<float>(n_offset+c_offset+pl) *= scale->data<float>(c);
            }
        }
    }
}

} // namespace cpu
} // namespace op
} // namespace eutopia
