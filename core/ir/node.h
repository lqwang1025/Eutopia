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

#include "op/ops_param.h"

namespace eutopia {

namespace op {
class Operator;
}

namespace core {
namespace ir {

class Tensor;
class Graph;
class Node final {
    friend class graph;
public:
    Node(Graph* graph);
    Node(Graph* graph, const op::BaseParam* param);
    ~Node(void);
    void setup(const op::BaseParam* param);
    const std::string& get_name(void) const;
    void set_name(const std::string& name);
    int32_t get_index(void) const;
    void set_index(const int32_t index);
    const std::string& get_op_type(void) const;
    void set_op_type(const std::string& op_type);
    void set_output_shape(const std::vector<uint32_t>& output_shape);
    const std::vector<uint32_t>& get_output_shape(void) const;
    void add_producer(const std::string& producer);
    const std::vector<std::string>& get_producers(void) const;
    void add_consumer(const std::string& consumer);
    const std::vector<std::string>& get_consumers(void) const;
    void set_is_first_node(bool is_input);
    void set_is_last_node(bool is_output);
    bool is_first_node(void) const;
    bool is_last_node(void) const;
    void set_dynamic_shape(bool dynamic_shape);
    bool dynamic_shape(void) const;
    void set_is_sparse(bool is_sparse);
    bool is_sparse(void) const;
    void set_is_quantize(bool is_quantize);
    bool is_quantize(void) const;
    void set_in_place(bool in_inplace);
    bool in_place(void) const;
    void set_weight_shared(bool weight_shared);
    bool weight_shared(void) const;
    
    void set_weights(const std::vector<Tensor*>& weights);
    std::vector<Tensor*> get_weights(void) const;
    
    void set_bias(const std::vector<Tensor*>& bias);
    std::vector<Tensor*> get_bias(void) const;
    
    bool is_trainning(void) const;
    void set_is_trainning(bool is_trainning);
    const Tensor* get_output_tensor(void) const;
    void set_graph(Graph* graph);
    const Graph* get_graph(void) const;
    void dump(void);
    void forward(const std::vector<const Tensor*> input_tensors);
    void backward(void);
    void update(void);
    void run(void);
private:
    op::Operator* op_;
    std::string name_;
    int32_t index_; // squence index
    std::string op_type_;
    bool is_trainning_;
    bool is_quantize_;
    bool weight_shared_;
    bool is_sparse_;
    bool is_first_node_;
    bool is_last_node_;
    bool dynamic_shape_;
    bool in_place_;
    std::string device_;
    std::vector<uint32_t> output_shape_;
    std::vector<std::string> producers_;
    std::vector<std::string> consumers_;
    std::vector<Tensor*> weights_;
    std::vector<Tensor*> biases_;
    Tensor* output_tensor_;
    Tensor* diff_tensor_;
    Graph* graph_; // The graph that owned this node.
private:
    Node& operator=(Node&);
    Node(const Node&);
};

} // namespace ir
} // namespace core
} // namespace eutopia

#endif /* __NODE_H__ */

