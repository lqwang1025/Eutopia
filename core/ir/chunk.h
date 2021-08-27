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
 * File       : chunk.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-08:18:27:41
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __CHUNK_H__
#define __CHUNK_H__

#include <stdint.h>

namespace eutopia {
namespace core {
namespace ir {

class Tensor;

class Chunk final {
public:
    Chunk(Tensor *owner);
    ~Chunk();
    Chunk(Tensor *owner, uint32_t byte_size, void *data = nullptr);
    void set_data(uint32_t byte_size, void *data = nullptr);
    uint32_t get_byte_size() const;

    template <typename T> T& at(uint32_t index);
    
    const void* get_data_ptr() const;
    void* get_mutable_data_ptr();
    
protected:
    Tensor *owner_;
    uint32_t byte_size_;
    void *data_;
    bool own_data_;

protected:
    void _release_data();
};

} // namespace ir
} // namespace core
} // namespace eutopia

#endif /* __CHUNK_H__ */
