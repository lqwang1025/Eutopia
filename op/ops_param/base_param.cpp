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
 * File       : base_param.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-20:09:39:37
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include "op/ops_param/base_param.h"

namespace eutopia {
namespace op {

void BaseParam::copy_from(const BaseParam* param) {
    sparse        = param->sparse;
    quantize      = param->quantize;
    weight_shared = param->weight_shared;
    first_op      = param->first_op;
    last_op       = param->last_op;
    op_type       = param->op_type;
    op_name       = param->op_name;
    device        = param->device;
    CHECK(EUTOPIA_SUPPORT_OPS.count(op_type) != 0, "Unsupport operator type.");
}

} // namespace op
} // namespace eutopia
