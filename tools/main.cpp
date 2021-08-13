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
 * File       : main.cc
 * Authors    : Daniel Wang
 * Create Time: 2021-08-08:13:30:59
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <iostream>
#include <cstdlib>

#include "core/data_type.h"
#include "core/ir/tensor.h"
#include "core/ir/node.h"
#include "core/ir/graph.h"
// #include "op/op_register.h"
#include "op/ops_param/convolution2D_param.h"

using namespace std;

int main(int argc, char** argv) {
    // eutopia::core::ir::Tensor tensor;
    // std::vector<uint32_t> dims = {224, 224, 3};
    // tensor.set_data(dims, eutopia::core::DataType::EUTOPIA_DT_UINT8);
    // const uint8_t& s = tensor.data<uint8_t>({112, 112, 2});
    // std::cout<<(int)s<<std::endl;
    // tensor.mutable_data<uint8_t>({112, 112, 2}) = 255;
    // eutopia::core::ir::Graph g;
    // std::cout<<(int)s<<std::endl;
    eutopia::core::ir::Node node;
    struct eutopia::op::Convolution2DParam conv_param;
    conv_param.pad_type = "SAME";
    node.setup_op(&conv_param);
    return 0;
}
