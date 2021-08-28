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
 * File       : proto_interface.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-28:18:27:38
 * Email      : wangliquan21@qq.com
 * Description:
 */
#include <fstream>

#include "op/cpu/convolution2D.h"
#include "op/cpu/fully_connected.h"
#include "op/cpu/input.h"
#include "op/cpu/pooling.h"

#include "core/ir/graph.h"
#include "core/ir/node.h"
#include "core/framework/node_def.pb.h"
#include "core/framework/graph_def.pb.h"
#include "core/framework/attr_value.pb.h"
#include "core/logging.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

namespace eutopia {
namespace core {
namespace ir {

eutopia::framework::NodeDef* Node::_conv2d_proto() const {
    eutopia::framework::NodeDef* conv2d_node = new eutopia::framework::NodeDef;
    auto conv2d_attr = conv2d_node->mutable_attr();
    conv2d_node->set_op(get_op_type());
    conv2d_node->set_name(get_name());
    eutopia::framework::AttrValue attr_value;
    attr_value.set_b(is_quantize_);
    conv2d_attr->insert({"is_quantize", attr_value});
    attr_value.set_b(weight_shared_);
    conv2d_attr->insert({"weight_shared", attr_value});
    attr_value.set_b(dynamic_shape_);
    conv2d_attr->insert({"dynamic_shape", attr_value});
    attr_value.set_b(is_last_node_);
    conv2d_attr->insert({"is_last_node", attr_value});
    attr_value.set_b(is_sparse_);
    conv2d_attr->insert({"is_sparse", attr_value});
    attr_value.set_b(is_first_node_);
    conv2d_attr->insert({"is_first_node", attr_value});
    attr_value.set_b(in_place_);
    conv2d_attr->insert({"in_place", attr_value});
    attr_value.clear_b();
    attr_value.set_s(device_);
    conv2d_attr->insert({"device", attr_value});
    
    eutopia::framework::AttrValue_ListValue list_value;
    list_value.clear_i();
    for (auto it : output_shape_) {
        list_value.add_i(it);
    }
    attr_value.clear_s();
    *attr_value.mutable_list() = list_value;
    conv2d_attr->insert({"output_shape", attr_value});
    
    op::cpu::Convolution2DOperator* conv2d_op = static_cast<op::cpu::Convolution2DOperator*>(op_);
    op::Convolution2DParam* op_param = conv2d_op->op_param_;
    list_value.clear_i();
    for (auto it : op_param->kernel_shape) {
        list_value.add_i(it);
    }
    *attr_value.mutable_list() = list_value;
    conv2d_attr->insert({"kernels", attr_value});
    list_value.clear_i();
    for (auto it : op_param->strides) {
        list_value.add_i(it);
    }
    *attr_value.mutable_list() = list_value;
    conv2d_attr->insert({"strides", attr_value});
    
    list_value.clear_i();
    for (auto it : op_param->dilations) {
        list_value.add_i(it);
    }
    *attr_value.mutable_list() = list_value;
    conv2d_attr->insert({"dilations", attr_value});

    list_value.clear_i();
    for (auto it : op_param->pads) {
        list_value.add_i(it);
    }
    *attr_value.mutable_list() = list_value;
    conv2d_attr->insert({"pads", attr_value});
    attr_value.clear_list();
    attr_value.set_s(op_param->pad_type);
    conv2d_attr->insert({"padding", attr_value});
    
    for (auto it : get_producers()) {
        conv2d_node->add_input(it);
    }
    return conv2d_node;
}

eutopia::framework::NodeDef* Node::_fc_proto() const {
    eutopia::framework::NodeDef* fc_node = new eutopia::framework::NodeDef;
    auto fc_attr = fc_node->mutable_attr();
    fc_node->set_op(get_op_type());
    fc_node->set_name(get_name());
    eutopia::framework::AttrValue attr_value;
    attr_value.set_b(is_quantize_);
    fc_attr->insert({"is_quantize", attr_value});
    attr_value.set_b(weight_shared_);
    fc_attr->insert({"weight_shared", attr_value});
    attr_value.set_b(dynamic_shape_);
    fc_attr->insert({"dynamic_shape", attr_value});
    attr_value.set_b(is_last_node_);
    fc_attr->insert({"is_last_node", attr_value});
    attr_value.set_b(is_sparse_);
    fc_attr->insert({"is_sparse", attr_value});
    attr_value.set_b(is_first_node_);
    fc_attr->insert({"is_first_node", attr_value});
    attr_value.set_b(in_place_);
    fc_attr->insert({"in_place", attr_value});
    attr_value.clear_b();
    attr_value.set_s(device_);
    fc_attr->insert({"device", attr_value});
    
    eutopia::framework::AttrValue_ListValue list_value;
    list_value.clear_i();
    for (auto it : output_shape_) {
        list_value.add_i(it);
    }
    attr_value.clear_s();
    *attr_value.mutable_list() = list_value;
    fc_attr->insert({"output_shape", attr_value});
    
    op::cpu::FullyConnectedOperator* fc_op = static_cast<op::cpu::FullyConnectedOperator*>(op_);
    op::FullyConnectedParam* op_param = fc_op->op_param_;
    attr_value.clear_list();
    attr_value.set_i(op_param->num_outputs);
    fc_attr->insert({"num_outputs", attr_value});
    
    for (auto it : get_producers()) {
        fc_node->add_input(it);
    }
    return fc_node;
}

eutopia::framework::NodeDef* Node::_input_proto() const {
    eutopia::framework::NodeDef* input_node = new eutopia::framework::NodeDef;
    auto input_attr = input_node->mutable_attr();
    input_node->set_op(get_op_type());
    input_node->set_name(get_name());
    eutopia::framework::AttrValue attr_value;
    attr_value.set_b(is_quantize_);
    input_attr->insert({"is_quantize", attr_value});
    attr_value.set_b(weight_shared_);
    input_attr->insert({"weight_shared", attr_value});
    attr_value.set_b(dynamic_shape_);
    input_attr->insert({"dynamic_shape", attr_value});
    attr_value.set_b(is_last_node_);
    input_attr->insert({"is_last_node", attr_value});
    attr_value.set_b(is_sparse_);
    input_attr->insert({"is_sparse", attr_value});
    attr_value.set_b(is_first_node_);
    input_attr->insert({"is_first_node", attr_value});
    attr_value.set_b(in_place_);
    input_attr->insert({"in_place", attr_value});
    attr_value.clear_b();
    attr_value.set_s(device_);
    input_attr->insert({"device", attr_value});
    
    eutopia::framework::AttrValue_ListValue list_value;
    list_value.clear_i();
    for (auto it : output_shape_) {
        list_value.add_i(it);
    }
    attr_value.clear_s();
    *attr_value.mutable_list() = list_value;
    input_attr->insert({"output_shape", attr_value});
    
    op::cpu::InputOperator* input_op = static_cast<op::cpu::InputOperator*>(op_);
    op::InputParam* op_param = input_op->op_param_;
    attr_value.clear_list();
    attr_value.set_f(op_param->mean);
    input_attr->insert({"mean", attr_value});
    
    attr_value.set_f(op_param->std);
    input_attr->insert({"std", attr_value});
    
    for (auto it : get_producers()) {
        input_node->add_input(it);
    }
    return input_node;
}

eutopia::framework::NodeDef* Node::_pooling_proto() const {
    eutopia::framework::NodeDef* pool_node = new eutopia::framework::NodeDef;
    auto pool_attr = pool_node->mutable_attr();
    pool_node->set_op(get_op_type());
    pool_node->set_name(get_name());
    eutopia::framework::AttrValue attr_value;
    attr_value.set_b(is_quantize_);
    pool_attr->insert({"is_quantize", attr_value});
    attr_value.set_b(weight_shared_);
    pool_attr->insert({"weight_shared", attr_value});
    attr_value.set_b(dynamic_shape_);
    pool_attr->insert({"dynamic_shape", attr_value});
    attr_value.set_b(is_last_node_);
    pool_attr->insert({"is_last_node", attr_value});
    attr_value.set_b(is_sparse_);
    pool_attr->insert({"is_sparse", attr_value});
    attr_value.set_b(is_first_node_);
    pool_attr->insert({"is_first_node", attr_value});
    attr_value.set_b(in_place_);
    pool_attr->insert({"in_place", attr_value});
    attr_value.clear_b();
    attr_value.set_s(device_);
    pool_attr->insert({"device", attr_value});
    
    eutopia::framework::AttrValue_ListValue list_value;
    list_value.clear_i();
    for (auto it : output_shape_) {
        list_value.add_i(it);
    }
    attr_value.clear_s();
    *attr_value.mutable_list() = list_value;
    pool_attr->insert({"output_shape", attr_value});
    
    op::cpu::PoolingOperator* pool_op = static_cast<op::cpu::PoolingOperator*>(op_);
    op::PoolingParam* op_param = pool_op->op_param_;
    list_value.clear_i();
    for (auto it : op_param->kernels) {
        list_value.add_i(it);
    }
    *attr_value.mutable_list() = list_value;
    pool_attr->insert({"kernels", attr_value});

    list_value.clear_i();
    for (auto it : op_param->strides) {
        list_value.add_i(it);
    }
    *attr_value.mutable_list() = list_value;
    pool_attr->insert({"strides", attr_value});
    
    attr_value.clear_list();
    attr_value.set_s(op_param->pool_type);
    pool_attr->insert({"pool_type", attr_value});
    for (auto it : get_producers()) {
        pool_node->add_input(it);
    }
    return pool_node;
}

eutopia::framework::NodeDef* Node::to_proto() const {
    if (proto_map_.count(op_type_) != 0) {
        return (this->*proto_map_.at(op_type_))();
    } else {
        return nullptr;
    }
    
}

void Graph::to_proto() const {
    framework::GraphDef* graph = new framework::GraphDef;
    graph->set_name(get_name());
    eutopia::framework::AttrValue attr_value;
    attr_value.set_s("Daniel Wang");
    graph->mutable_attr()->insert({"author", attr_value});
    for (int i = 0; i < (int)seq_nodes_.size(); ++i) {
        Node* cur_node = seq_nodes_[i];
        framework::NodeDef* node = cur_node->to_proto();
        framework::NodeDef* g_node = graph->add_node();
        *g_node = *node;
        delete node;
    }
    std::fstream file("out.pb", std::ios::out | std::ios::trunc | std::ios::binary);
    if (!graph->SerializeToOstream(&file)) {
        EU_WARN<<"Failed to serialize."<<RESET;
   }
    std::string ddd = graph->DebugString();
    std::cout<<"debu:"<<ddd<<std::endl;
    file.close();
    delete graph;
}

} // namespace ir
} // namespace core
} // namespace eutopia

