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
 * File       : convolution2d.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-12:07:50:53
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include "op/op.h"

namespace eutopia {
namespace op {

DECLARE_OPERATOR(Convolution2d);

void Convolution2d::infer_shape(const std::vector<core::ir::Tensor*> input_tensors, std::vector<uint32_t>& output_shape) {
    
}

void Convolution2d::forward(const std::vector<core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) {
    
}

void Convolution2d::backward(const std::vector<core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) {
    
}

REGISTER_OPERATOR(EConvolution2d, Convolution2d);

} // namespace op
} // namespace eutopia

