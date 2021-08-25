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

#include "op/cpu/convolution2D.h"
#include "op/ops_param.h"
#include "core/ir/node.h"

namespace eutopia {
namespace op {
namespace cpu {

Convolution2DOperator::Convolution2DOperator(const BaseParam* op_param) {
    Convolution2DParam* conv_param = new Convolution2DParam;
    conv_param->copy_from(op_param);
    op_param_ = conv_param;
}

void Convolution2DOperator::infer_shape(const InputShapes& input_shapes, std::vector<uint32_t>& output_shape) {
    core::ir::Tensor* weight = node_->get_weight();
}

void Convolution2DOperator::forward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* output_tensor) {
    
}

void Convolution2DOperator::backward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* output_tensor) {
    
}

} // namespace cpu
} // namespace op
} // namespace eutopia

