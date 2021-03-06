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

namespace core {
namespace ir {
class Node;
}
}

namespace op {

struct BaseParam;


class Operator {
    friend class core::ir::Node;
public:
    using InputShapes = std::vector< std::vector<uint32_t> >;
    Operator(){};
    virtual ~Operator(){};
    virtual void infer_shape(const InputShapes& input_shapes, std::vector<uint32_t>& output_shape) = 0;
    virtual void forward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) = 0;
    virtual void backward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) = 0;
    void set_node(const core::ir::Node* node);
protected:
    const core::ir::Node* node_ = nullptr;
private:
    Operator(const Operator&);
    Operator& operator=(const Operator&);
};

class Holder {
public:
    typedef Operator* (*Creator)(const BaseParam* op_param);
    typedef std::map<std::string, Creator> OpCreatorMap;
    
    static OpCreatorMap& new_op_creator() {
        static OpCreatorMap* op_creator_map = new OpCreatorMap();
        return *op_creator_map;
    }
    
    static void add_op_crreator(const std::string& op_type, Creator creator) {
        OpCreatorMap& op_creator_map = new_op_creator();
        if (op_creator_map.count(op_type) == 1) {
            EU_WARN << op_type << " had already been registered. "<<EU_ENDL;
            return;
        }
        op_creator_map[op_type] = creator;
    }
    
    static Creator get_op_creator(const std::string& op_type) {
        OpCreatorMap& op_creator_map = new_op_creator();
        CHECK(op_creator_map.count(op_type) != 0, "This op have not been register.");
        return op_creator_map[op_type];
    }
    
private:
    Holder();
    Holder(const Holder&);
    Holder& operator=(const Holder&);
};

class ToHolder {
public:
    typedef Operator* (*Creator)(const BaseParam* op_param);
    ToHolder(const std::string& op_type, Creator creator) {
        Holder::add_op_crreator(op_type, creator);
    }
};

#define REGISTER_OP_CREATOR(op_type, creator)                       \
    static ToHolder g_creator_operator_##op_type(#op_type, creator)


#define REGISERT_OP_CLASS(op_type, op)                                  \
    static Operator* Creator_##op_type(const BaseParam* op_param) {     \
        return new op(op_param);                                        \
    }                                                                   \
    REGISTER_OP_CREATOR(op_type, Creator_##op_type)

#define DECLARE_OPERATOR(param_type, sub_class)                         \
    class sub_class : public Operator {                                 \
        friend class core::ir::Node;                                    \
    public:                                                             \
    sub_class(const BaseParam* op_param);                               \
    virtual ~sub_class(){ delete op_param_; }                           \
    virtual void infer_shape(const InputShapes& input_shapes, std::vector<uint32_t>& output_shape) override; \
    virtual void forward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) override; \
    virtual void backward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* Output_tensor) override; \
    private:                                                            \
    param_type* op_param_;                                              \
    sub_class();                                                        \
    sub_class(const sub_class&);                                        \
    sub_class& operator=(const sub_class&);                             \
    }

} // namespace op
} // namespace eutopia

#endif /* __OP_H__ */
