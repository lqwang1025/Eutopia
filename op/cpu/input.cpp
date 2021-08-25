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
 * File       : input.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-23:20:09:19
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include "op/cpu/input.h"
#include "op/ops_param.h"

namespace eutopia {
namespace op {
namespace cpu {

InputOperator::InputOperator(const BaseParam* op_param) {
    InputParam* input_param = new InputParam;
    input_param->copy_from(op_param);
    op_param_ = input_param;
}

void InputOperator::infer_shape(const std::vector<const core::ir::Tensor*> input_tensors, std::vector<uint32_t>& output_shape) {
    CHECK(input_tensors.size() == 1, "Input node only have one input.");
    const core::ir::Tensor* tensor = input_tensors[0];
    output_shape = tensor->dims();
    // std::vector<uint32_t> input_dims = op_param_->input_dims;
}

void InputOperator::forward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* output_tensor) {
    CHECK(input_tensors.size() == 1, "Input node only have one input.");
    const core::ir::Tensor* input_tensor = input_tensors[0];
    *output_tensor = *input_tensor;
}

void InputOperator::backward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* output_tensor) {
    CHECK(input_tensors.size() == 1, "Input node only have one input.");
    const core::ir::Tensor* input_tensor = input_tensors[0];
    *output_tensor = *input_tensor;
}

} // namespace cpu
} // namespace op
} // namespace eutopia

