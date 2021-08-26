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
 * File       : pooling.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-24:10:04:08
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <cmath>

#include "op/cpu/pooling.h"
#include "op/ops_param.h"

namespace eutopia {
namespace op {
namespace cpu {

PoolingOperator::PoolingOperator(const BaseParam* op_param) {
    PoolingParam* pool_param = new PoolingParam;
    pool_param->copy_from(op_param);
    op_param_ = pool_param;
}

void PoolingOperator::infer_shape(const InputShapes& input_shapes, std::vector<uint32_t>& output_shape) {
    CHECK(input_shapes.size()==1, "Now pooling only support 1 input.");
    std::vector<uint32_t> input_shape = input_shapes[0]; // n c h w
    std::vector<uint32_t> kernels = op_param_->kernels; // h w
    std::vector<uint32_t> strides = op_param_->strides; // h w
    CHECK(input_shape.size()!=0,);
    std::vector<uint32_t> pads = {0, 0}; // todo : add pads in parmeter
    
    uint32_t pooled_h = static_cast<int32_t>(std::ceil(static_cast<float>(input_shape[2] + 2*pads[0] - kernels[0]) / strides[0])) + 1;
    uint32_t pooled_w = static_cast<int32_t>(std::ceil(static_cast<float>(input_shape[3] + 2*pads[1] - kernels[1]) / strides[1])) + 1;
    output_shape = {input_shape[0], input_shape[1], pooled_h, pooled_w};
}

void PoolingOperator::forward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) {
    
}

void PoolingOperator::backward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) {
    
}

} // namespace cpu
} // namespace op
} // namespace eutopia

