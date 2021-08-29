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
 * File       : fully_connected.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-24:11:44:00
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include "op/ops_param.h"
#include "op/cpu/fully_connected.h"

#include "core/ir/tensor.h"
#include "core/ir/node.h"

#include "op/cpu/compute/gemm.h"

namespace eutopia {
namespace op {
namespace cpu {

FullyConnectedOperator::FullyConnectedOperator(const BaseParam* op_param) {
    FullyConnectedParam* fc_param = new FullyConnectedParam;
    fc_param->copy_from(op_param);
    op_param_ = fc_param;
}

void FullyConnectedOperator::infer_shape(const InputShapes& input_shapes, std::vector<uint32_t>& output_shape) {
    CHECK(input_shapes.size()==1, "Now conv2d only support 1 input.");
    std::vector<uint32_t> input_shape = input_shapes[0]; // n c h w
    uint32_t num_outputs = op_param_->num_outputs;
    output_shape = {input_shape[0], 1, 1, num_outputs};
}

void FullyConnectedOperator::forward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* output_tensor) {
    CHECK(input_tensors.size()==1,);
    core::ir::Tensor* input_tensor = const_cast<core::ir::Tensor*>(input_tensors[0]);
    const std::vector<uint32_t> ori_input_shape = input_tensor->dims();
    input_tensor->reshape({ori_input_shape[0], ori_input_shape[1]*ori_input_shape[2]*ori_input_shape[3], 1}); // flatten
    std::vector<uint32_t> input_shape = input_tensor->dims();
    std::vector<uint32_t> output_shape = node_->get_output_shape();
    output_tensor->set_data(output_shape, input_tensor->get_data_type());
    
    const core::ir::Tensor* weight = node_->get_weight();
    const core::ir::Tensor* bias = node_->get_bias();
    std::vector<uint32_t> weight_shape = weight->dims();
    gemm(weight, input_tensor, output_tensor);
    input_tensor->reshape(ori_input_shape);
}

void FullyConnectedOperator::backward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* output_tensor) {
    
}

} // namespace cpu
} // namespace op
} // namespace eutopia

