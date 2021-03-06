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
 * File       : fully_connected_param.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-23:14:53:15
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __FULLY_CONNECTED_PARAM_H__
#define __FULLY_CONNECTED_PARAM_H__
#include <vector>
#include <string>

#include "op/ops_param/base_param.h"

namespace eutopia {
namespace op {

struct FullyConnectedParam : public BaseParam {
    FullyConnectedParam() {
        op_type = FULLYCONNECTED;
    }
    uint32_t num_outputs;
    std::string activation = "None";
    bool with_bias = true;
    bool with_batch_norm = false;
    std::vector<float> mean;
    std::vector<float> var;
    std::vector<float> gamma;
    std::vector<float> beta;
    float epsilon;
    void copy_from(const BaseParam* param);
    virtual ~FullyConnectedParam() {}
};

typedef struct FullyConnectedParam FullyConnectedParam;

} // namespace op
} // namespace eutopia

#endif /* __FULLY_CONNECTED_PARAM_H__ */

