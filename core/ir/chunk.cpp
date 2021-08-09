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
 * File       : chunk.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-08:18:39:44
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <stdlib.h>

#include <iostream>
#include "core/ir/chunk.h"
#include "core/ir/tensor.h"

namespace eutopia {
namespace core {
namespace ir {

Chunk::Chunk() {
    byte_size_ = 0;
    data_      = nullptr;
    own_data_  = false;
    owner_     = nullptr;
}

Chunk::Chunk(Tensor* owner, uint32_t byte_size, void* data) {
    set_data(owner, byte_size, data);
}

Chunk::~Chunk() {
    _release_data();
}

void Chunk::set_data(Tensor* owner, uint32_t byte_size, void* data) {
    if (data == nullptr) {
        data_ = malloc(byte_size);//todo:check data is not null
        own_data_ = true;
    } else {
        _release_data();
        data_ = data;
        own_data_ = false;
    }
    byte_size_ = byte_size;
    owner_ = owner;
}

uint32_t Chunk::get_byte_size() const {
    return byte_size_;
}

const void* Chunk::get_data_ptr() const {
    return data_;
}

void* Chunk::get_mutable_data_ptr() const {
    return data_;
}

void Chunk::_release_data() {
    if (own_data_ && data_ != nullptr) {
        free(data_);
    }
}

template <typename T>
const T& Chunk::operator[](const uint32_t index) const {
    uint8_t data_type_bytes = get_data_type_byte(owner_->data_type_);
    void* data = data_;
    return ((T*)((uint8_t*)data + index * data_type_bytes))[0];
}

template <typename T>
T& Chunk::operator[](const uint32_t index) {
    void* data = data_;
    uint8_t data_type_bytes = get_data_type_byte(owner_->data_type_);
    return ((T*)((uint8_t*)data + index * data_type_bytes))[0];
}

template const uint8_t& Chunk::operator[]<uint8_t>(const uint32_t index) const;
template const int8_t& Chunk::operator[]<int8_t>(const uint32_t index) const;
template const int16_t& Chunk::operator[]<int16_t>(const uint32_t index) const;
template const uint16_t& Chunk::operator[]<uint16_t>(const uint32_t index) const;
template const uint32_t& Chunk::operator[]<const uint32_t>(const uint32_t index) const;
template const int32_t& Chunk::operator[]<int32_t>(const uint32_t index) const;
template const float& Chunk::operator[]<float>(const uint32_t index) const;

template uint8_t& Chunk::operator[]<uint8_t>(const uint32_t index);
template int8_t& Chunk::operator[]<int8_t>(const uint32_t index);
template int16_t& Chunk::operator[]<int16_t>(const uint32_t index);
template uint16_t& Chunk::operator[]<uint16_t>(const uint32_t index);
template uint32_t& Chunk::operator[]<uint32_t>(const uint32_t index);
template int32_t& Chunk::operator[]<int32_t>(const uint32_t index);
template float& Chunk::operator[]<float>(const uint32_t index);

} // namespace ir
} //namespace eutopia
} //namespace core
