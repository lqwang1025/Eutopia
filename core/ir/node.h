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
#include <cstdint>

#include "op/op.h"

namespace eutopia {
namespace core {
namespace ir {

class Tensor;

class Node final {
public:
    Node();
    ~Node();
    const std::string& get_name() const;
    void set_name(std::string& name);
    uint16_t get_index() const;
    void set_index(const uint16_t index);
    const std::string& get_op_type() const;
    void set_op_type(const std::string& op_type);
    void set_output_shape(const std::vector<uint32_t>& output_shape);
    const std::vector<uint32_t>& get_output_shape() const;
    void add_producer(const std::string& producer);
    const std::vector<std::string>& get_producers() const;
    void add_consumer(const std::string& consumer);
    const std::vector<std::string>& get_consumers() const;
    void set_is_first_node(bool is_input);
    void set_is_last_node(bool is_output);
    bool is_first_node() const;
    bool is_last_node() const;
    void set_dynamic_shape(bool dynamic_shape);
    bool dynamic_shape() const;
    void set_is_sparse(bool is_sparse);
    bool is_sparse() const;
    void set_is_quantize(bool is_quantize);
    bool is_quantize() const;
    void set_in_place(bool in_inplace);
    bool in_place() const;
    void set_with_bias(bool with_bias);
    bool with_bias() const;
    void set_weight_shared(bool weight_shared);
    bool weight_shared() const;
    bool is_trainning() const;
    void set_is_trainning(bool is_trainning);
    const Tensor* get_output_tensor() const;
    void dump();
    void forward();
    void backward();
    void update();
    void run();
private:
    op::Operator* op_;
    std::string name_;
    uint16_t index_; // squence index
    std::string op_type_;
    std::vector<uint32_t> output_shape_;
    bool is_trainning_;
    bool is_quantize_;
    bool weight_shared_;
    bool is_sparse_;
    bool is_first_node_;
    bool is_last_node_;
    bool dynamic_shape_;
    bool in_place_;
    bool with_bias_;
    std::vector<std::string> producers_;
    std::vector<std::string> consumers_;
    std::vector<Tensor*> weights_;
    std::vector<Tensor*> biases_;
    Tensor* output_tensor_;
private:
    Node& operator=(Node&){}
    Node(const Node&){}
};

} // namespace ir
} // namespace core
} // namespace eutopia

#endif /* __NODE_H__ */

