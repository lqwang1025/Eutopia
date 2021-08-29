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
 * Create Time: 2021-08-12:16:35:16
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __BASE_PARAM_H__
#define __BASE_PARAM_H__

#include <string>

#include "op/ops_name.h"
#include "core/logging.h"

namespace eutopia {
namespace op {
    
struct BaseParam {
    BaseParam() {}
    bool sparse = false;
    bool quantize = false;
    bool weight_shared = false;
    bool first_op = false;
    bool last_op  = false;
    std::string device = "cpu";
    std::string op_type;
    std::string op_name;
    void copy_from(const BaseParam* param);
    virtual ~BaseParam() {}
};

typedef struct BaseParam BaseParam;

} // namespace op
} // namespace eutopia

#endif /* __BASE_PARAM_H__ */
