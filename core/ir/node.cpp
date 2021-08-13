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
 * File       : node.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-09:22:02:38
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include "core/ir/node.h"
#include "op/op.h"
#include "op/base_param.h"

#include <iostream>

namespace eutopia {
namespace core {
namespace ir {

Node::Node() {
    set_is_trainning(false);
    set_dynamic_shape(false);
}

Node::Node(const struct op::BaseParam* param): Node() {
    setup_op(param);
}

void Node::setup_op(const struct op::BaseParam* param) {
    set_op_type(param->op_type);
    set_name(param->op_name);
    set_is_sparse(param->sparse);
    set_is_quantize(param->quantize);
    set_weight_shared(param->weight_shared);
    set_is_first_node(param->first_op);
    set_is_last_node(param->last_op);
    op_ = op::Holder::get_op_creator(op_type_)(param);
}

const std::string& Node::get_name() const {
    return name_;
}

void Node::set_name(const std::string& name) {
    name_ = name;
}

uint16_t Node::get_index() const {
    return index_;
}

void Node::set_index(const uint16_t index) {
    index_ = index;
}

const std::string& Node::get_op_type() const {
    return op_type_;
}

void Node::set_op_type(const std::string& op_type) {
    op_type_ = op_type;
}

void Node::set_output_shape(const std::vector<uint32_t>& output_shape) {
    output_shape_ = output_shape;
}

const std::vector<uint32_t>& Node::get_output_shape() const {
    return output_shape_;
}

void Node::add_producer(const std::string& producer) {
    producers_.push_back(producer);
}

const std::vector<std::string>& Node::get_producers() const {
    return producers_;
}

void Node::add_consumer(const std::string& consumer) {
    consumers_.push_back(consumer);
}

const std::vector<std::string>& Node::get_consumers() const {
    return consumers_;
}

void Node::set_is_first_node(bool is_first_node) {
    is_first_node_ = is_first_node;
}

void Node::set_is_last_node(bool is_last_node) {
    is_last_node_ = is_last_node;
}

bool Node::is_last_node() const {
    return is_last_node_;
}

bool Node::is_first_node() const {
    return is_first_node_;
}

void Node::set_dynamic_shape(bool dynamic_shape) {
    dynamic_shape_ = dynamic_shape;
}

bool Node::dynamic_shape() const {
    return dynamic_shape_;
}

void Node::set_is_sparse(bool is_sparse) {
    is_sparse_ = is_sparse;
}

bool Node::is_sparse() const {
    return is_sparse_;
}

void Node::set_is_quantize(bool is_quantize) {
    is_quantize_ = is_quantize;
}

bool Node::is_quantize() const {
    return is_quantize_;
}

void Node::set_in_place(bool in_place) {
    in_place_ = in_place;
}

bool Node::in_place() const {
    return in_place_;
}

void Node::set_weight_shared(bool weight_shared) {
    weight_shared_ = weight_shared;
}

bool Node::weight_shared() const {
    return weight_shared_;
}

bool Node::is_trainning() const {
    return is_trainning_;
}

void Node::set_is_trainning(bool is_trainning) {
    is_trainning_ = is_trainning;
}

const Tensor* Node::get_output_tensor() const {
    return output_tensor_;
}

void Node::forward() {
    
}

void Node::backward() {
    
}

void Node::update() {
    
}

void Node::run() {
    //todo
}

void Node::dump() {
    
}

Node::~Node() {
    
}

} // namespace ir
} // namespace core
} // namespace eutopia
