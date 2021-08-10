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
 * File       : graph.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-09:22:04:27
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <map>
#include <string>
#include <vector>

namespace eutopia {
namespace core {
namespace ir {

class Node;
class Tensor;
class Graph final {
public:
    Graph();
    ~Graph();
    void forward();
    void backward();
    void update();
    void run();
    void dump();
    void set_name(const std::string& name);
    const std::string& get_name() const;
    void add_node(Node* node);
    Node* get_node(const std::string& node_name) const;
    
    void add_input_tensor(Tensor* tensor);
    Tensor* get_input_tensor(const std::string& tensor_name) const;
    
    void add_output_tensor(Tensor* tensor);
    Tensor* get_output_tensor(const std::string& tensor_name) const;
    
    void add_tensor(Tensor* tensor);
    std::vector<Tensor*>& get_all_tensor() const;
    bool is_trainning() const;
    void set_is_trainning(bool is_trainning);
private:
    bool is_trainning_;
    std::string name_;
    std::map<std::string, Tensor*> all_tensors_;
    std::map<std::string, Tensor*> input_tensors_;
    std::map<std::string, Tensor*> output_tensors_;
    std::vector<Node*> nodes_;
    std::map<std::string, Node*> nodes_map_;
};

} // namespace ir
} // namespace core
} // namespace eutopia

#endif /* __GRAPH_H__ */

