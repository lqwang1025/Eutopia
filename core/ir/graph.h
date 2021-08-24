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

#include <unordered_map>
#include <string>
#include <vector>

#include "op/ops_param.h"
#include "core/ir/node.h"

namespace eutopia {
namespace core {
namespace ir {

class Tensor;
struct BaseParam;

class Graph final {
public:
    Graph(void);
    ~Graph(void);
    
    struct TrainParam {
        int batch;
        int epoch;
        std::string optimize_type;
        std::string loss_type;
    };
    
    void forward(void);
    void backward(void);
    void update(void);
    void run(void);
    void dump(void);
    void set_name(const std::string& name);
    const std::string& get_name(void) const;
    
    Node* add_node(op::BaseParam* param = nullptr);
    Node* get_node(const std::string& node_name) const;
    
    void remove_node(Node* node);//todo
    
    void add_input_tensor(Tensor* tensor);//todo
    Tensor* get_input_tensor(const std::string& tensor_name) const;//todo
    
    void add_output_tensor(Tensor* tensor);//todo
    Tensor* get_output_tensor(const std::string& node_name) const;//todo

    void sort_by_execute(void);
    
    bool is_trainning(void) const;
    void set_is_trainning(bool is_trainning);
private:
    bool is_trainning_;
    std::string name_;
    TrainParam train_param_;
    std::vector<Node*> input_nodes_;
    std::vector<Node*> output_nodes_;
    std::vector<Node*> seq_nodes_;
    std::vector<Node*> own_nodes_;
    std::unordered_map<std::string, Node*> node_name_map_;
    std::unordered_map<std::string, Tensor*> input_tensors_;
    std::unordered_map<std::string, Tensor*> output_tensors_;
private:
    void _top_sort(std::unordered_map< std::string, std::vector<std::string> >& name_producers);
    Graph& operator=(Graph&);
    Graph(Graph&);
};

} // namespace ir
} // namespace core
} // namespace eutopia

#endif /* __GRAPH_H__ */

