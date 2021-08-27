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
 * File       : gemm.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-26:16:49:20
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <vector>

#include "op/cpu/compute/gemm.h"
#include "core/ir/tensor.h"
#include "core/logging.h"

namespace eutopia {
namespace op {
namespace cpu {

void gemm(const core::ir::Tensor* weight, core::ir::Tensor* data_col,
          const core::ir::Tensor* bias, const core::ir::Tensor* result) {
    std::vector<uint32_t> weight_shape = weight->dims();
    CHECK(weight_shape.size()==4,);
    std::vector<uint32_t> data_shape = data_col->dims();
    CHECK(data_shape.size()==2,);
    uint32_t weight_w = weight_shape[3]*weight_shape[2]*weight_shape[1];
    uint32_t H = weight_shape[0];
    uint32_t W = data_shape[1];
    uint32_t K = data_shape[0];
    int h, w , k;
#pragma omp parallel for
    for (h = 0; h < H; ++h) {
        for (k = 0; k < K; ++k) {
            register float A_PART = weight->data<float>(h*weight_w+k);
            float sum = 0.;
            for (w = 0; w < W; ++w) {
                sum /*C[i*ldc+j]*/ += A_PART*data_col->data<float>(k*data_shape[1]+w);
                std::cout<<sum;
            }
            std::cout<<std::endl;
        }
    }
    std::cout<<"debug:"<<H<<" "<<W<<" "<<K<<std::endl;
}

} // namespace cpu
} // namespace op
} // namespace eutopia
