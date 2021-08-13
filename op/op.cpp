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
 * File       : op.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-12:07:50:16
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include "op/op.h"
#include "core/logging.h"
#include "core/ir/tensor.h"
#include "op/base_param.h"

namespace eutopia {
namespace op {

void Operator::infer_shape(const std::vector<core::ir::Tensor*> input_tensors, std::vector<uint32_t>& output_shape) {
    
    return;
}

void Operator::forward(const std::vector<core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) {
    return;
}

void Operator::backward(const std::vector<core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) {
    return;
}

} // namespace op
} // namespace eutopia
