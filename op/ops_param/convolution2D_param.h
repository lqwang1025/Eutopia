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
 * Create Time: 2021-08-13:10:58:56
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __CONVOLUTION2D_PARAM_H__
#define __CONVOLUTION2D_PARAM_H__

#include <vector>
#include <string>
#include <set>

#include "op/ops_param/base_param.h"
#include "core/logging.h"

namespace eutopia {
namespace op {

static const std::set<std::string> SUPPORT_PAD_TYPES = {"VALID", "SAME"};

struct Convolution2DParam : public BaseParam {
    Convolution2DParam() {
        op_type = CONVOLUTION2D;
    }
    std::vector<uint32_t> kernel_shape; // h w ic oc
    std::vector<uint32_t> stride;
    std::vector<uint32_t> dilations;
    std::vector<int32_t> pads;
    std::string pad_type;
    uint32_t group;
    void copy_from(const struct BaseParam* param);
    virtual ~Convolution2DParam() {}
};

typedef struct Convolution2DParam Convolution2DParam;

} // namespace op
} // namespace eutopia

#endif /* __CONVOLUTION2D_PARAM_H__ */

