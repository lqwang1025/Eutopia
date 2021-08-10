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
 * File       : node.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-09:22:02:34
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __NODE_H__
#define __NODE_H__

#include <string>
#include <vector>

namespace eutopia {
namespace core {
namespace ir {

class Tensor;

class Node final {
pubilc:
    Node();
    ~Node();
    const std::string& get_name() const;
    void set_name(std::string& name);
    const uint16_t& get_index() const;
    void set_index(uint16_t index);
    const std::string& get_op_type() const;
    void set_op_type(std::string& op_type);
    void set_output_shape(std::vector<uint32_t>& output_shape);
    const std::vector<uint32_t>& get_output_shape() const;
    void set_producer(std::string& producer);
    const std::vector<std::string>& get_producers() const;
    void set_consumer(std::string& consumer);
    const std::vector<std::string>& get_consumers() const;
    void set_is_input(bool is_input);
    void set_is_output(bool is_output);
    bool is_input() const;
    bool is_output() const;
    void run();
private:
    std::string name_;
    uint16_t index_;
    std::string op_type_;
    std::vector<uint32_t> output_shape_;
    bool is_quantize_;
    bool weight_shared_;
    bool is_sparse_;
    bool is_input_;
    bool is_output_;
    bool dynamic_shape_;
    bool in_place_;
    bool with_bias_;
    std::vector<std::string> producers_;
    std::vector<std::string> consumers_;
    std::vector<Tensor*> weights_;
    std::vector<Tensor*> biases_;
    std::vector<Tensor*> inputs_;
    Tensor* output_;
};

} // namespace ir
} // namespace core
} // namespace eutopia


#endif /* __NODE_H__ */

