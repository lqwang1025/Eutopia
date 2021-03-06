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
 * File       : im2col.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-26:16:49:20
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __IM2COL_H__
#define __IM2COL_H__

#include <vector>
#include <cstdint>

#include "op/ops_param.h"

namespace eutopia {
namespace core {
namespace ir {
class Tensor;
}
}
namespace op {
namespace cpu {

void im2col(const std::vector<uint32_t>& output_shape, const Convolution2DParam* op_param, const core::ir::Tensor* im, core::ir::Tensor& col);

void col2im(const Convolution2DParam* op_param, const core::ir::Tensor* col, core::ir::Tensor& im);

} // namespace cpu
} // namespace op
} // namespace eutopia

#endif /* __IM2COL_H__ */

