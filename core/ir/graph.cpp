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

#include <vector>
#include <cstdio>

#include "core/ir/graph.h"
#include "core/ir/node.h"
#include "core/ir/tensor.h"
#include "core/logging.h"

namespace eutopia {
namespace core {
namespace ir {

Graph::Graph() {
    
}

void Graph::forward() {
    for (int i = 0; i < seq_nodes_.size(); ++i) {
        Node* cur_node = seq_nodes_[i];
        std::vector<const Tensor*> cur_inputs(0);
        const std::vector<std::string>& cur_producers = cur_node->get_producers();
        std::string cur_name = cur_node->get_name();
        if (cur_producers.size() == 0) {
            CHECK(input_tensors_.count(cur_name)!=0, "Please set inputs to graph.");
            cur_inputs.push_back(input_tensors_[cur_name]);
        } else {
            for (auto& it : cur_producers) {
                CHECK(node_name_map_.count(it)!=0, "Do not find the Node.");
                const Node* papa = node_name_map_[it];
                cur_inputs.push_back(papa->get_output_tensor());
            }
        }
        cur_node->forward(cur_inputs);
    }
}

void Graph::backward() {
    
}

void Graph::update() {
    
}

void Graph::run() {
    
}

void Graph::dump() {
    
}

Node* Graph::add_node(op::BaseParam* param) {
    Node* node = nullptr;
    if (param != nullptr) {
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

void Graph::_top_sort(std::unordered_map< std::string, std::vector<std::string> >& name_producers) {
    if (name_producers.size() == 0) return;
    for (auto& it : name_producers) {
        if (it.second.size() == 0) {
            std::string name = it.first;
            name_producers.erase(name);
            CHECK(node_name_map_.count(name) != 0, "Do not find your node.");
            seq_nodes_.push_back(node_name_map_[name]);
            for (auto& _it : name_producers) {
                for (int i = 0; i < _it.second.size(); ++i) {
                    if (_it.second[i] == name) {
                        _it.second.erase(_it.second.begin()+i);
                    }
                }
            }
            break;
        }
    }
    _top_sort(name_producers);
}

void Graph::sort_by_execute(void) {
    std::unordered_map< std::string, std::vector<std::string> > name_producers;
    for (int i = 0; i < (int)own_nodes_.size(); ++i) {
        Node* node = own_nodes_[i];
        name_producers[node->get_name()] = node->get_producers();
        node_name_map_[node->get_name()] = node;
    }
    _top_sort(name_producers);
    for (int  i = 0; i < seq_nodes_.size(); ++i) {
        Node* node = seq_nodes_[i];
        node->set_index(i);
        std::string cur_name = node->get_name();
        for (int _i = i; _i < seq_nodes_.size(); ++_i) {
            Node* next_node = seq_nodes_[_i];
            for (auto it : next_node->get_producers()) {
                if (it == cur_name) {
                    node->add_consumer(next_node->get_name());
                }
            }
        }
        if (node->get_producers().size() == 0) {
            input_nodes_.push_back(node);
        }
        if (node->get_consumers().size() == 0) {
            output_nodes_.push_back(node);
            node->set_is_last_node(true);
        }
    }
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
