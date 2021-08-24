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
 * File       : input.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-23:21:32:34
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __INPUT_H__
#define __INPUT_H__

#include "op/op.h"
#include "op/ops_param.h"

namespace eutopia {
namespace op {
namespace cpu {

DECLARE_OPERATOR(InputParam, InputOperator);

} // namespace cpu
} // namespace op
} // namespace eutopia

#endif /* __INPUT_H__ */

