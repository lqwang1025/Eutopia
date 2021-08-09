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

#include <cstdarg>
#include <iostream>

#include "core/logging.h"
#include "core/ir/tensor.h"
#include "core/ir/chunk.h"

namespace eutopia {
namespace core {
namespace ir {

const static uint8_t MAX_DIMS_SIZE = 32;

Tensor::Tensor() {
    data_type_ = DataType::EUTOPIA_DT_UNKNOWN;
    byte_size_ = 0;
    mem_       = new Chunk;
}

Tensor::Tensor(const std::vector<int>& dims, DataType data_type, void* mem) {
    set_data(dims, data_type, mem);
}

void Tensor::set_data(const std::vector<int>& dims, DataType data_type, void* mem) {
    CHECK(dims.size() != 0) << "Please make sure your dims size is not 0.";
    CHECK(dims.size() <= MAX_DIMS_SIZE)
        << "Please make sure your dims size is not greater than " << MAX_DIMS_SIZE;
    dims_ = dims;
    int elem_num = 1;
    for (auto& it : dims) {
        elem_num *= it;
    }
    byte_size_ = (uint32_t)(get_data_type_byte(data_type)*elem_num);
    mem_->set_data(this, byte_size_, mem);
}

uint8_t Tensor::dims_size() const {
    return (uint8_t)dims_.size();
}

void Tensor::set_name(const std::string& name) {
    name_ = name;
}

const std::string& Tensor::get_name() const {
    return name_;
}

template<typename T>
const T& Tensor::data(const std::vector<uint32_t>& indices) const {
    if (indices.size() != dims_size()) {
        std::cout<<"err:"<<std::endl;
    }
    std::vector<uint32_t> strides(dims_size(), 1);
    uint32_t index = 0;
    for (int i = 0; i < (int)indices.size(); ++i) {
        if (dims_[i] <= indices[i]) {
            std::cout<<"err:"<<std::endl;
        }
        for (int _i = i+1; _i < indices.size(); ++_i) {
            strides[i] *= dims_[_i];
        }
        index += strides[i]*indices[i];
    }
    const T& value = mem_->at<T>(index);
    return value;
}

template<typename T>
T& Tensor::mutable_data(const std::vector<uint32_t>& indices) {
    if (indices.size() != dims_size()) {
        std::cout<<"err:"<<std::endl;
    }
    std::vector<uint32_t> strides(dims_size(), 1);
    uint32_t index = 0;
    for (int i = 0; i < (int)indices.size(); ++i) {
        if (dims_[i] <= indices[i]) {
            std::cout<<"err:"<<std::endl;
        }
        for (int _i = i+1; _i < indices.size(); ++_i) {
            strides[i] *= dims_[_i];
        }
        index += strides[i]*indices[i];
    }
    T& value = mem_->at<T>(index);
    return value;
}

template const uint8_t& Tensor::data(const std::vector<uint32_t>& dims) const;
template const int8_t& Tensor::data(const std::vector<uint32_t>& dims) const;
template const int16_t& Tensor::data(const std::vector<uint32_t>& dims) const;
template const uint16_t& Tensor::data(const std::vector<uint32_t>& dims) const;
template const uint32_t& Tensor::data(const std::vector<uint32_t>& dims) const;
template const int32_t& Tensor::data(const std::vector<uint32_t>& dims) const;
template const float& Tensor::data(const std::vector<uint32_t>& dims) const;

template uint8_t& Tensor::mutable_data(const std::vector<uint32_t>& dims);
template int8_t& Tensor::mutable_data(const std::vector<uint32_t>& dims);
template int16_t& Tensor::mutable_data(const std::vector<uint32_t>& dims);
template uint16_t& Tensor::mutable_data(const std::vector<uint32_t>& dims);
template uint32_t& Tensor::mutable_data(const std::vector<uint32_t>& dims);
template int32_t& Tensor::mutable_data(const std::vector<uint32_t>& dims);
template float& Tensor::mutable_data(const std::vector<uint32_t>& dims);

Tensor::~Tensor() {
    delete mem_;
}

} // namespace ir
} // namespace eutopia
} // namespace core
