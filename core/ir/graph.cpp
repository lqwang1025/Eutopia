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
 * File       : graph.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-09:22:04:36
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include "core/ir/graph.h"
#include "core/ir/node.h"
#include "core/logging.h"

namespace eutopia {
namespace core {
namespace ir {

Graph::Graph() {
    
}

Graph::~Graph() {
    
}

void Graph::forward() {
    
}

void Graph::backward() {
    
}

void Graph::update() {
    
}

void Graph::run() {
    
}

void Graph::dump() {
    
}

void Graph::add_node(Node* node) {
    const std::string& node_name = node->get_name();
    if (nodes_map_.count(node_name) != 0) {
        EU_WARN<<node_name<<" "<<"has alreadly in graph."<<EU_ENDL;
    }
    nodes_map_[node_name] = node;
}

Node* Graph::get_node(const std::string& node_name) const {
    CHECK(nodes_map_.count(node_name) != 0, "Do not find your node in graph.");
    return nodes_map_.find(node_name)->second;
}

void Graph::add_input_tensor(Tensor* tensor) {
    input_tensors_.push_back(tensor);
}

std::vector<Tensor*>& Graph::get_input_tensors() const {
    return input_tensors_;
}

void Graph::add_output_tensor(Tensor* tensor) {
    output_tensors_.push_back(tensor);
}

std::vector<Tensor*>& Graph::get_output_tensors() const {
    return output_tensors_;
}

void Graph::add_tensor(Tensor* tensor) {
    all_tensors_.push_back(tensor);
}

std::vector<Tensor*>& Graph::get_all_tensor() const {
    return all_tensors_;
}

bool Graph::is_trainning() const {
    return is_trainning_;
}

void Graph::set_is_trainning(bool is_trainning) {
    is_trainning_ = is_trainning;
}

void Graph::set_name(const std::string& name) {
    name_ = name;
}

const std::string& Graph::get_name() const {
    return name_;
}

} // namespace ir
} // namespace core
} // namespace eutopia
