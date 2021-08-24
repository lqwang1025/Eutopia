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
 * File       : op_register.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-23:21:33:14
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __OP_REGISTER_H__
#define __OP_REGISTER_H__

#include "op/cpu/input.h"
#include "op/cpu/convolution2D.h"
#include "op/cpu/fully_connected.h"
#include "op/cpu/pooling.h"

namespace eutopia {
namespace op {
namespace cpu {

REGISERT_OP_CLASS(Input, InputOperator);
REGISERT_OP_CLASS(Convolution2D, Convolution2DOperator);
REGISERT_OP_CLASS(Pooling, PoolingOperator);
REGISERT_OP_CLASS(FullyConnected, FullyConnectedOperator);

} // namespace cpu
} // namespace op
} // namespace eutopia

#endif /* __OP_REGISTER_H__ */

