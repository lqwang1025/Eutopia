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


#include <cstdio>
namespace eutopia {
namespace core {
namespace ir {

Graph::Graph(void) {
    
}

void Graph::forward(void) {
    
}

void Graph::backward(void) {
    
}

void Graph::update(void) {
    
}

void Graph::run(void) {
    
}

void Graph::dump(void) {
    
}

Node* Graph::add_node(op::BaseParam* param) {
    Node* node = nullptr;
    if (param == nullptr) {
        node = new Node(this, param);
    } else {
        node = new Node(this);
    }
    CHECK(node!=nullptr, "Graph add node failed.");
    own_nodes_.push_back(node);
    return node;
}

Node* Graph::get_node(const std::string& node_name) const {
    int len = node_name.size();
    for (int i = 0; i < own_nodes_.size(); ++i) {
        Node* node = own_nodes_[i];
        const std::string& target_name = node->get_name();
        int start = target_name.size() - len;
        if (start) continue;
        return node;
    }
    return nullptr;
}

void Graph::sort_by_execute(void) {
    
}

Tensor* Graph::get_output_tensor(const std::string& node_name) const {
    CHECK(output_tensors_.count(node_name) != 0, "Do not find your node in graph.");
    return output_tensors_.find(node_name)->second;
}

bool Graph::is_trainning(void) const {
    return is_trainning_;
}

void Graph::set_is_trainning(bool is_trainning) {
    is_trainning_ = is_trainning;
}

void Graph::set_name(const std::string& name) {
    name_ = name;
}

const std::string& Graph::get_name(void) const {
    return name_;
}

Graph::~Graph(void) {
    for (int i = 0; i < (int)own_nodes_.size(); ++i) {
        if (own_nodes_[i]->get_graph() == this) {
            delete own_nodes_[i];
        }
    }
}

} // namespace ir
} // namespace core
} // namespace eutopia
