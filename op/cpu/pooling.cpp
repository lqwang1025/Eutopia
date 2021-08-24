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

void PoolingOperator::infer_shape(const std::vector<core::ir::Tensor*> input_tensors, std::vector<uint32_t>& output_shape) {
    
}

void PoolingOperator::forward(const std::vector<core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) {
    
}

void PoolingOperator::backward(const std::vector<core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) {
    
}

} // namespace cpu
} // namespace op
} // namespace eutopia
