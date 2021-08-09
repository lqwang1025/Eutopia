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
 * File       : tensor.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-08:10:23:09
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <math.h>
#include <iostream>
#include "core/ir/tensor.h"
#include "core/ir/chunk.h"

namespace eutopia {
namespace core {
namespace ir {

Tensor::Tensor() {
    data_type_ = DataType::EUTOPIA_DT_UNKNOWN;
    byte_size_ = 0;
    mem_       = new Chunk;
}

Tensor::Tensor(std::vector<int>& dims, DataType data_type, void* mem) {
    set_data(dims, data_type, mem);
}

void Tensor::set_data(std::vector<int>& dims, DataType data_type, void* mem) {
    if (dims.size() == 0) return;//todo log here
    int elem_num = 1;
    for (auto& it : dims) {
        elem_num *= it;
    }
    
    byte_size_ = (uint32_t)(get_data_type_byte(data_type)*elem_num);
    mem_->set_data(byte_size_, mem);
}

void Tensor::set_name(const std::string& name) {
    name_ = name;
}

const std::string& Tensor::get_name() const {
    return name_;
}

Tensor::~Tensor() {
    delete mem_;
}

} // namespace ir
} // namespace eutopia
} // namespace core
