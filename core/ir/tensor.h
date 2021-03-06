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
 * File       : tensor.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-08:10:22:59
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <vector>
#include <string>
#include <cstdarg>

#include "core/data_type.h"

namespace onnx {
class TensorProto;
}

namespace eutopia {
namespace core {
namespace ir {

class Chunk;

class Tensor final {
    friend class Chunk;
public:
    Tensor();
    Tensor(const std::vector<uint32_t>& dims, DataType data_type, void* mem=nullptr);
    ~Tensor();
    void set_data(const std::vector<uint32_t>& dims, DataType data_type, void* mem=nullptr);
    void set_name(const std::string& name);
    const std::string& name() const;
    
    template<typename T>
    const T& data(const std::vector<uint32_t>& indices) const;
    
    template<typename T>
    T& mutable_data(const std::vector<uint32_t>& indices);

    template<typename T>
    const T& data(uint32_t index) const;
    
    template<typename T>
    T& mutable_data(uint32_t index);
    
    uint8_t dims_size() const;
    void set_dims(const std::vector<uint32_t>& dims);
    const std::vector<uint32_t>& dims() const;
    void reshape(const std::vector<uint32_t>& shape);
    DataType get_data_type() const;
    void set_data_type(const DataType& data_type);
    Tensor& operator=(const Tensor& rhs);
    uint32_t byte_size() const;
    uint32_t total() const;
    Chunk* chunk() const;
    void to_proto(onnx::TensorProto* tensor_proto);
private:
    std::string name_;
    DataType data_type_;
    std::vector<uint32_t> dims_;
    uint32_t byte_size_;
    Chunk* mem_;
};

} // namespace ir
} // namespace core
} // namespace eutopia

#endif /* __TENSOR_H__ */

