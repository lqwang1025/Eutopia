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
 * File       : op.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-12:07:50:10
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __OP_H__
#define __OP_H__

#include <map>
#include <string>
#include <vector>

#include "core/logging.h"
#include "core/ir/tensor.h"

namespace eutopia {
namespace op {

class Operator {
public:
    Operator()=default;
    virtual ~Operator()=default;
    virtual void infer_shape(const std::vector<core::ir::Tensor*> input_tensors, std::vector<uint32_t>& output_shape);
    virtual void forward(const std::vector<core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor);
    virtual void backward(const std::vector<core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor);
};

class Holder {
public:
    using OperatorMap = std::map<std::string, Operator*>;
    static OperatorMap& new_operator() {
        static OperatorMap* operator_map = new OperatorMap();
        return *operator_map;
    }
    
    static void add_operator(const std::string& op_type, Operator* op) {
        OperatorMap& operator_map = new_operator();
        if (operator_map.count(op_type) == 1) {
            EU_WARN << op_type << " had already been registered. "<<EU_ENDL;
            return;
        }
        operator_map[op_type] = op;
    }
    
    static Operator* get_operator(const std::string& op_type) {
        OperatorMap& operator_map = new_operator();
        if (operator_map.count(op_type) == 0) {
            return nullptr;
        }
        return operator_map[op_type];
    }
    
    static void release() {
        OperatorMap& operator_map = new_operator();
        for (auto& it : operator_map) {
            delete it.second;
        }
        delete &operator_map;
    }
private:
    Holder() {}
    Holder& operator=(Holder&) {}
};

class ToHolder {
public:
    ToHolder(const std::string& op_type, Operator* op) {
        Holder::add_operator(op_type, op);
    }
};

#define REGISTER_OPERATOR(op_type, operator)                           \
    static ToHolder g_creator_operator_##op_type(#op_type, new operator)

#define DECLARE_OPERATOR(sub_class)                                     \
    class sub_class : public Operator {                                 \
    public:                                                             \
    sub_class() = default;                                              \
    virtual void infer_shape(const std::vector<core::ir::Tensor*> input_tensors, std::vector<uint32_t>& output_shape); \
    virtual void forward(const std::vector<core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor); \
    virtual void backward(const std::vector<core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor); \
    }

} // namespace op
} // namespace eutopia

#endif /* __OP_H__ */

