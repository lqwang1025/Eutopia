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
 * File       : pooling_param.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-21:12:52:23
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __POOLING_PARAM_H__
#define __POOLING_PARAM_H__
#include <vector>
#include <string>
#include <set>

#include "op/ops_param/base_param.h"

namespace eutopia {
namespace op {

static const std::set<std::string> SUPPORT_POOL_TYPES = {"MAX", "MEAN"};

struct PoolingParam : public BaseParam {
    PoolingParam() {
        op_type = POOLING;
    }
    std::vector<uint32_t> kernels; // h w
    std::vector<uint32_t> strides;
    std::string pool_type; // SAME
    void copy_from(const BaseParam* param);
    virtual ~PoolingParam() {}
};

typedef struct PoolingParam PoolingParam;

} // namespace op
} // namespace eutopia

#endif /* __POOLING_PARAM_H__ */

