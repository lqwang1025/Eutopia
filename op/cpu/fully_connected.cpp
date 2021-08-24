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

namespace eutopia {
namespace op {
namespace cpu {

FullyConnectedOperator::FullyConnectedOperator(const BaseParam* op_param) {
    FullyConnectedParam* fc_param = new FullyConnectedParam;
    fc_param->copy_from(op_param);
    op_param_ = fc_param;
}

void FullyConnectedOperator::infer_shape(const std::vector<const core::ir::Tensor*> input_tensors, std::vector<uint32_t>& output_shape) {
    std::cout<<"fc"<<std::endl;
}

void FullyConnectedOperator::forward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) {
    
}

void FullyConnectedOperator::backward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) {
    
}

} // namespace cpu
} // namespace op
} // namespace eutopia

