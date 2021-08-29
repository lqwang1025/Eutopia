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
#include <string>

#include "op/cpu/convolution2D.h"
#include "op/cpu/fully_connected.h"
#include "op/cpu/input.h"
#include "op/cpu/pooling.h"

#include "core/ir/graph.h"
#include "core/ir/node.h"
#include "core/framework/onnx.pb.h"
#include "core/logging.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

namespace eutopia {
namespace core {
namespace ir {

using onnx::AttributeProto_AttributeType;

static void _add_attr_to_node(onnx::NodeProto* node, const std::string& name, const float& value) {
    onnx::AttributeProto* attr = node->add_attribute();
    attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
    attr->set_name(name);
    attr->set_f(value);
}

static void _add_attr_to_node(onnx::NodeProto* node, const std::string& name, const int& value) {
    onnx::AttributeProto* attr = node->add_attribute();
    attr->set_type(onnx::AttributeProto_AttributeType_INT);
    attr->set_name(name);
    attr->set_i(value);
}

static void _add_attr_to_node(onnx::NodeProto* node, const std::string& name, const std::string& value) {
    onnx::AttributeProto* attr = node->add_attribute();
    attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    attr->set_name(name);
    attr->set_s(value);
}

static void _add_attr_to_node(onnx::NodeProto* node, const std::string& name, const std::vector<float>& value) {
    onnx::AttributeProto* attr = node->add_attribute();
    attr->set_type(onnx::AttributeProto_AttributeType_FLOATS);
    attr->set_name(name);
    for (auto it : value) {
        attr->add_floats(it);
    }
}

static void _add_attr_to_node(onnx::NodeProto* node, const std::string& name, const std::vector<int>& value) {
    onnx::AttributeProto* attr = node->add_attribute();
    attr->set_type(onnx::AttributeProto_AttributeType_INTS);
    attr->set_name(name);
    for (auto it : value) {
        attr->add_ints(it);
    }
}

static void _add_attr_to_node(onnx::NodeProto* node, const std::string& name, const std::vector<uint32_t>& value) {
    onnx::AttributeProto* attr = node->add_attribute();
    attr->set_type(onnx::AttributeProto_AttributeType_INTS);
    attr->set_name(name);
    for (auto it : value) {
        attr->add_ints((int32_t)it);
    }
}

static void _add_attr_to_node(onnx::NodeProto* node, const std::string& name, const std::vector<std::string>& value) {
    onnx::AttributeProto* attr = node->add_attribute();
    attr->set_type(onnx::AttributeProto_AttributeType_STRINGS);
    attr->set_name(name);
    for (auto it : value) {
        attr->add_strings(it);
    }
}

onnx::NodeProto* Node::_conv2d_proto() const {
    onnx::NodeProto* conv2d_node = new onnx::NodeProto;
    conv2d_node->set_op_type(get_op_type());
    conv2d_node->set_name(get_name());
    conv2d_node->add_output(get_name());
    
    _add_attr_to_node(conv2d_node, "is_quantize", (int)is_quantize_);
    _add_attr_to_node(conv2d_node, "weight_shared", (int)weight_shared_);
    _add_attr_to_node(conv2d_node, "dynamic_shape", (int)dynamic_shape_);
    _add_attr_to_node(conv2d_node, "is_last_node", (int)is_last_node_);
    _add_attr_to_node(conv2d_node, "is_sparse", (int)is_sparse_);
    _add_attr_to_node(conv2d_node, "is_first_node", (int)is_first_node_);
    _add_attr_to_node(conv2d_node, "in_place", (int)in_place_);
    _add_attr_to_node(conv2d_node, "device", device_);
    _add_attr_to_node(conv2d_node, "output_shape", output_shape_);
    
    op::cpu::Convolution2DOperator* conv2d_op = static_cast<op::cpu::Convolution2DOperator*>(op_);
    op::Convolution2DParam* op_param = conv2d_op->op_param_;
    _add_attr_to_node(conv2d_node, "kennels", op_param->kernel_shape);
    _add_attr_to_node(conv2d_node, "strides", op_param->strides);
    _add_attr_to_node(conv2d_node, "dilations", op_param->dilations);
    _add_attr_to_node(conv2d_node, "pads", op_param->pads);
    _add_attr_to_node(conv2d_node, "padding", op_param->pad_type);
    
    for (auto it : get_producers()) {
        conv2d_node->add_input(it);
    }
    
    return conv2d_node;
}

onnx::NodeProto* Node::_fc_proto() const {
    onnx::NodeProto* fc_node = new onnx::NodeProto;
    fc_node->set_op_type(get_op_type());
    fc_node->set_name(get_name());
    fc_node->add_output(get_name());
    
    _add_attr_to_node(fc_node, "is_quantize", (int)is_quantize_);
    _add_attr_to_node(fc_node, "weight_shared", (int)weight_shared_);
    _add_attr_to_node(fc_node, "dynamic_shape", (int)dynamic_shape_);
    _add_attr_to_node(fc_node, "is_last_node", (int)is_last_node_);
    _add_attr_to_node(fc_node, "is_sparse", (int)is_sparse_);
    _add_attr_to_node(fc_node, "is_first_node", (int)is_first_node_);
    _add_attr_to_node(fc_node, "in_place", (int)in_place_);
    _add_attr_to_node(fc_node, "device", device_);
    _add_attr_to_node(fc_node, "output_shape", output_shape_);
    
    op::cpu::FullyConnectedOperator* fc_op = static_cast<op::cpu::FullyConnectedOperator*>(op_);
    op::FullyConnectedParam* op_param = fc_op->op_param_;
    _add_attr_to_node(fc_node, "num_outputs", (int)op_param->num_outputs);
    
    
    for (auto it : get_producers()) {
        fc_node->add_input(it);
    }
    
    return fc_node;
}

onnx::NodeProto* Node::_input_proto() const {
    onnx::NodeProto* input_node = new onnx::NodeProto;
    input_node->set_op_type(get_op_type());
    input_node->set_name(get_name());
    input_node->add_output(get_name());
    
    _add_attr_to_node(input_node, "is_quantize", (int)is_quantize_);
    _add_attr_to_node(input_node, "weight_shared", (int)weight_shared_);
    _add_attr_to_node(input_node, "dynamic_shape", (int)dynamic_shape_);
    _add_attr_to_node(input_node, "is_last_node", (int)is_last_node_);
    _add_attr_to_node(input_node, "is_sparse", (int)is_sparse_);
    _add_attr_to_node(input_node, "is_first_node", (int)is_first_node_);
    _add_attr_to_node(input_node, "in_place", (int)in_place_);
    _add_attr_to_node(input_node, "device", device_);
    _add_attr_to_node(input_node, "output_shape", output_shape_);
    
    op::cpu::InputOperator* input_op = static_cast<op::cpu::InputOperator*>(op_);
    op::InputParam* op_param = input_op->op_param_;
    _add_attr_to_node(input_node, "mean", op_param->mean);
    _add_attr_to_node(input_node, "std", op_param->std);
    
    
    for (auto it : get_producers()) {
        input_node->add_input(it);
    }
    return input_node;
}

onnx::NodeProto* Node::_pooling_proto() const {
    onnx::NodeProto* pool_node = new onnx::NodeProto;
    pool_node->set_op_type(get_op_type());
    pool_node->set_name(get_name());
    pool_node->add_output(get_name());
    
    _add_attr_to_node(pool_node, "is_quantize", (int)is_quantize_);
    _add_attr_to_node(pool_node, "weight_shared", (int)weight_shared_);
    _add_attr_to_node(pool_node, "dynamic_shape", (int)dynamic_shape_);
    _add_attr_to_node(pool_node, "is_last_node", (int)is_last_node_);
    _add_attr_to_node(pool_node, "is_sparse", (int)is_sparse_);
    _add_attr_to_node(pool_node, "is_first_node", (int)is_first_node_);
    _add_attr_to_node(pool_node, "in_place", (int)in_place_);
    _add_attr_to_node(pool_node, "device", device_);
    _add_attr_to_node(pool_node, "output_shape", output_shape_);
    
    op::cpu::PoolingOperator* pool_op = static_cast<op::cpu::PoolingOperator*>(op_);
    op::PoolingParam* op_param = pool_op->op_param_;
    _add_attr_to_node(pool_node, "kernels", op_param->kernels);
    _add_attr_to_node(pool_node, "strides", op_param->strides);
    _add_attr_to_node(pool_node, "pool_type", op_param->pool_type);
    
    for (auto it : get_producers()) {
        pool_node->add_input(it);
    }
    
    return pool_node;
}

onnx::NodeProto* Node::to_proto() const {
    if (proto_map_.count(op_type_) != 0) {
        return (this->*proto_map_.at(op_type_))();
    } else {
        return nullptr;
    }
}

void Graph::to_proto() const {
    onnx::ModelProto* model = new onnx::ModelProto;
    onnx::GraphProto* graph = new onnx::GraphProto;
    graph->set_name(get_name());
    for (int i = 0; i < (int)seq_nodes_.size(); ++i) {
        Node* cur_node = seq_nodes_[i];
        onnx::NodeProto* node = cur_node->to_proto();
        onnx::NodeProto* g_node = graph->add_node();
        *g_node = *node;
        delete node;
    }
    model->set_allocated_graph(graph);
    std::fstream file("out.onnx", std::ios::out | std::ios::trunc | std::ios::binary);
    if (!model->SerializeToOstream(&file)) {
        EU_WARN<<"Failed to serialize."<<RESET;
    }
    file.close();
    delete model;
}

} // namespace ir
} // namespace core
} // namespace eutopia

