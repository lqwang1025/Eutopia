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
 * File       : blas.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-26:16:49:20
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __GEMM_H__
#define __GEMM_H__

namespace eutopia {
namespace core {
namespace ir {
class Tensor;
}
}
namespace op {
namespace cpu {

void gemm(const core::ir::Tensor* weight, core::ir::Tensor* data_col, core::ir::Tensor* result);

void add_bias(const core::ir::Tensor* bias, core::ir::Tensor* result, const std::string& act_type);

void scale_result(const core::ir::Tensor* scale, core::ir::Tensor* result);

void scale_and_add_bias(const core::ir::Tensor* scale, const core::ir::Tensor* bias,
                        core::ir::Tensor* result, const std::string& act_type);

} // namespace cpu
} // namespace op
} // namespace eutopia

#endif /* __GEMM_H__ */
