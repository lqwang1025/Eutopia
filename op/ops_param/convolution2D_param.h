/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : convolution2D_param.h
 * Authors    : lqwang
 * Create Time: 2021-08-13:10:58:56
 * Description:
 *
 */

#ifndef __CONVOLUTION2D_PARAM_H__
#define __CONVOLUTION2D_PARAM_H__

#include <vector>
#include <string>
#include <set>

#include "op/base_param.h"
#include "core/logging.h"

namespace eutopia {
namespace op {

static const std::set<std::string> SUPPORT_PAD_TYPES = {"VALID", "SAME"};

struct Convolution2DParam : public BaseParam {
    Convolution2DParam() {
        op_type = CONVOLUTION2D;
    }
    std::vector<uint32_t> kernel_shape; // h w ic oc
    std::vector<uint32_t> stride;
    std::vector<uint32_t> dilations;
    std::vector<int32_t> pads;
    std::string pad_type;
    uint32_t group;
    void copy_from(const struct BaseParam* param) {
        this->BaseParam::copy_from(param);
        const struct Convolution2DParam* con_param = static_cast<const struct Convolution2DParam*>(param);
        kernel_shape = con_param->kernel_shape;
        stride       = con_param->stride;
        dilations    = con_param->dilations;
        pads         = con_param->pads;
        pad_type     = con_param->pad_type;
        group        = con_param->group;
        CHECK(SUPPORT_PAD_TYPES.count(pad_type)!=0, "Unsupport pad type.");
    }
};

} // namespace op
} // namespace eutopia

#endif /* __CONVOLUTION2D_PARAM_H__ */

