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
 * File       : convolution2d.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-12:07:50:53
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include "op/cpu/convolution2D.h"
#include "op/ops_param.h"
#include "core/ir/node.h"
#include "core/logging.h"
namespace eutopia {
namespace op {
namespace cpu {

Convolution2DOperator::Convolution2DOperator(const BaseParam* op_param) {
    Convolution2DParam* conv_param = new Convolution2DParam;
    conv_param->copy_from(op_param);
    op_param_ = conv_param;
}

void Convolution2DOperator::infer_shape(const InputShapes& input_shapes, std::vector<uint32_t>& output_shape) {
    CHECK(input_shapes.size()==1, "Now conv2d only support 1 input.");
    std::vector<uint32_t> input_shape = input_shapes[0]; // n h w c
    CHECK(input_shape.size()!=0,);
    std::vector<uint32_t> kernel_shape = op_param_->kernel_shape; // h w ic oc
    CHECK(kernel_shape.size()==4,);
    std::vector<uint32_t> strides      = op_param_->strides; // h w
    CHECK(strides.size()==2,);
    std::vector<uint32_t> dilations    = op_param_->dilations; // h w h w
    CHECK(dilations.size()==4,);
    std::string pad_type               = op_param_->pad_type;
    std::vector<int32_t> pads; // t b l r
    if (pad_type == "SAME") {
        int pad_h = ((input_shape[1]-1)*strides[0]+kernel_shape[0] - input_shape[1]) >> 1;
        int pad_w = ((input_shape[2]-1)*strides[1]+kernel_shape[1] - input_shape[2]) >> 1;
        pads = {pad_h, pad_h, pad_w, pad_w};
    } else if (pad_type == "VALID") {
        pads = op_param_->pads; // t b l r
    } else {
        CHECK(false, "Unsupport pad type.");
    }
    
    CHECK(pads.size()==4,);
    uint32_t kernel_extent_h           = dilations[0] * (kernel_shape[0] - 1) + 1;
    uint32_t kernel_extent_w           = dilations[1] * (kernel_shape[1] - 1) + 1;
    uint32_t output_h                  = (input_shape[1] + pads[0] + pads[1] - kernel_extent_h) / strides[0] + 1;
    uint32_t output_w                  = (input_shape[2] + pads[2] + pads[3] - kernel_extent_w) / strides[1] + 1;
    output_shape = {input_shape[0], output_h, output_w, kernel_shape[3]};
}

void Convolution2DOperator::forward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* output_tensor) {
    const core::ir::Tensor* weight = node_->get_weight();
    const core::ir::Tensor* bias = node_->get_bias();
    std::vector<uint32_t> dims_ = weight->dims();
    for (int h = 0; h < dims_[0]; ++h) {
        for (int w = 0; w < dims_[1]; ++w) {
            for (int ic = 0; ic < dims_[2]; ++ic) {
                for (int oc = 0; oc < dims_[3]; ++oc) {
                    std::cout<<weight->data<float>({h, w, ic, oc})<<" ";
                }
                std::cout<<std::endl;
            }
        }
    }
}

void Convolution2DOperator::backward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* output_tensor) {
    
}

} // namespace cpu
} // namespace op
} // namespace eutopia

