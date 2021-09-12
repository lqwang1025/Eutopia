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

#include "core/ir/node.h"
#include "core/logging.h"

#include "op/cpu/convolution2D.h"
#include "op/ops_param.h"
#include "op/cpu/compute/im2col.h"
#include "op/cpu/compute/blas.h"
#include "op/cpu/compute/batch_norm.h"

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
    std::vector<uint32_t> input_shape = input_shapes[0]; // n c h w
    CHECK(input_shape.size()!=0,);
    std::vector<uint32_t> kernel_shape = op_param_->kernel_shape; // oc ic h w
    CHECK(kernel_shape.size()==4,);
    std::vector<uint32_t> strides      = op_param_->strides; // h w
    CHECK(strides.size()==2,);
    std::vector<uint32_t> dilations    = op_param_->dilations; // h w h w
    CHECK(dilations.size()==4,);
    std::string pad_type               = op_param_->pad_type;
    std::vector<int32_t> pads; // t b l r
    if (pad_type == "SAME") {
        int pad_h = ((input_shape[2]-1)*strides[0]+kernel_shape[2] - input_shape[2]) >> 1;
        int pad_w = ((input_shape[3]-1)*strides[1]+kernel_shape[3] - input_shape[3]) >> 1;
        pads = {pad_h, pad_h, pad_w, pad_w};
    } else if (pad_type == "VALID") {
        pads = op_param_->pads; // t b l r
    } else {
        CHECK(false, "Unsupport pad type.");
    }
    
    CHECK(pads.size()==4,);
    uint32_t kernel_extent_h = dilations[0] * (kernel_shape[2] - 1) + 1;
    uint32_t kernel_extent_w = dilations[1] * (kernel_shape[3] - 1) + 1;
    uint32_t output_h        = (input_shape[2] + pads[0] + pads[1] - kernel_extent_h) / strides[0] + 1;
    uint32_t output_w        = (input_shape[3] + pads[2] + pads[3] - kernel_extent_w) / strides[1] + 1;
    output_shape             = {input_shape[0], kernel_shape[0], output_h, output_w};
}

void Convolution2DOperator::forward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* output_tensor) {
    CHECK(input_tensors.size()==1,"");
    std::vector<uint32_t> kernel_shape = op_param_->kernel_shape; // oc ic h w
    std::vector<uint32_t> strides = op_param_->strides; // h w
    uint32_t kernel_c = kernel_shape[1];
    uint32_t kernel_h = kernel_shape[2];
    uint32_t kernel_w = kernel_shape[3];
    uint32_t data_col_h  = kernel_c*kernel_w*kernel_h;
    std::vector<uint32_t> output_shape = node_->get_output_shape();
    CHECK(output_shape.size() == 4, );
    uint32_t output_h = output_shape[2];
    uint32_t output_w = output_shape[3];
    uint32_t data_col_w  = output_w*output_h;

    const core::ir::Tensor* input_tensor = input_tensors[0];
    uint32_t data_col_n = input_tensor->dims()[0];
    core::ir::Tensor data_col({data_col_n, data_col_h, data_col_w}, input_tensor->get_data_type());
    im2col(output_shape, op_param_, input_tensor, data_col);
    const core::ir::Tensor* weight = node_->get_weight();
    const core::ir::Tensor* bias = node_->get_bias();
    output_tensor->set_data(output_shape, input_tensor->get_data_type());
    gemm(weight, &data_col, output_tensor);
    if (op_param_->with_batch_norm) {
        if (node_->is_trainning()) {
            batch_norm_forward_training(op_param_->mean, op_param_->var, op_param_->gamma,
                                        op_param_->beta, op_param_->epsilon, output_tensor);
        } else {
            batch_norm_forward_inference(op_param_->mean, op_param_->var, op_param_->gamma,
                                         op_param_->beta, op_param_->epsilon, output_tensor);
        }
        
    } else {
        add_bias(bias, output_tensor, op_param_->activation);
    }
}

void Convolution2DOperator::backward(const std::vector<const core::ir::Tensor*> input_tensors, core::ir::Tensor* output_tensor) {
    
}

} // namespace cpu
} // namespace op
} // namespace eutopia

