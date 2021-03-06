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
 * File       : batch_norm.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-30:11:41:43
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <typeinfo>

#include "core/logging.h"
#include "core/ir/tensor.h"

#include "op/cpu/compute/batch_norm.h"

namespace eutopia {
namespace op {
namespace cpu {

void batch_norm_forward_training(std::vector<float>& mean, std::vector<float>& var,
                                 std::vector<float>& gamma, std::vector<float>& beta,
                                 float epsilon, core::ir::Tensor* output_tensor) {
    
}

void batch_norm_forward_inference(const std::vector<float>& mean, const std::vector<float>& var,
                                  const std::vector<float>& gamma, const std::vector<float>& beta,
                                  float epsilon, core::ir::Tensor* output_tensor) {
    std::vector<uint32_t> res_shape = output_tensor->dims(); // n c h w
    CHECK(res_shape.size()==4, );
    CHECK(mean.size()==res_shape[1], );
    CHECK(var.size()==res_shape[1], );
    CHECK(gamma.size()==res_shape[1], );
    CHECK(beta.size()==res_shape[1], );
    for (int n = 0; n < res_shape[0]; ++n) {
    }
    
}

} // namespace cpu
} // namespace op
} // namespace eutopia

