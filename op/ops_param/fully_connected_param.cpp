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
 * File       : fully_connected_param.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-23:14:57:42
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include "op/ops_param/fully_connected_param.h"

namespace eutopia {
namespace op {

void FullyConnectedParam::copy_from(const BaseParam* param) {
    this->BaseParam::copy_from(param);
    const FullyConnectedParam* fully_connected_param = static_cast<const FullyConnectedParam*>(param);
    num_outputs = fully_connected_param->num_outputs;
}
