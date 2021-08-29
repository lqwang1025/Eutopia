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
 * Create Time: 2021-08-20:09:42:02
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include "op/ops_param/convolution2D_param.h"

namespace eutopia {
namespace op {

void Convolution2DParam::copy_from(const BaseParam* param) {
    this->BaseParam::copy_from(param);
    const Convolution2DParam* con_param = static_cast<const Convolution2DParam*>(param);
    kernel_shape = con_param->kernel_shape;
    strides      = con_param->strides;
    dilations    = con_param->dilations;
    pads         = con_param->pads;
    pad_type     = con_param->pad_type;
    group        = con_param->group;
    activation   = con_param->activation;
    with_bias    = con_param->with_bias;
    CHECK(SUPPORT_PAD_TYPES.count(pad_type)!=0, "Unsupport pad type.");
}

} // namespace op
} // namespace eutopia
