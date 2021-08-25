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
 * File       : data_type.cpp
 * File       : filler.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-25:14:57:09
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include "core/ir/tensor.h"
#include "core/logging.h"
#include "utils/filler.h"
#include "utils/distributions.h"

namespace eutopia {
namespace utils {

Filler::Filler() {
    seed_ = false;
}

void Filler::set_seed(int seed) {
    seed_ = seed;
    have_seed_ = true;
}

void Filler::remove_seed() {
    have_seed_ = false;
}

int Filler::get_seed() const {
    return seed_;
}

void ConstantFiller::fill(core::ir::Tensor* tensor) {
    Distributions::uniform(tensor, constant_value_, constant_value_);
}

void UniformFiller::fill(core::ir::Tensor* tensor) {
    if (have_seed_) {
        Distributions::uniform(tensor, min_, max_, seed_);
    } else {
        Distributions::uniform(tensor, min_, max_);
    }
}

void NormalFiller::fill(core::ir::Tensor* tensor) {
    if (have_seed_) {
        Distributions::normal(tensor, mean_, std_, seed_);
    } else {
        Distributions::normal(tensor, mean_, std_);
    }
}

void TruncatedNormalFiller::fill(core::ir::Tensor* tensor) {
    if (have_seed_) {
        Distributions::truncated_normal(tensor, mean_, std_, seed_);
    } else {
        Distributions::truncated_normal(tensor, mean_, std_);
    }
}

void XavierFiller::fill(core::ir::Tensor* tensor) {
    CHECK(tensor->dims_size() == 4, "Xavier filler only dims 4.")
    float n;
    auto dims = tensor->dims();
    int fan_in = dims[1] * dims[2] * dims[3];
    int fan_out = dims[0] * dims[2] * dims[3];
    if (varianceType_ == VarianceType::FANIN) {
        n = fan_in;
    } else if (varianceType_ == VarianceType::FANOUT) {
        n = fan_out;
    } else {
        n = (fan_in + fan_out) / 2.0f;
    }

    float scale = sqrtf(3.0f / n);

    if (have_seed_) {
        Distributions::uniform(tensor, -scale, scale, seed_);
    } else {
        Distributions::uniform(tensor, -scale, scale);
    }
}

void MSRAFiller::fill(core::ir::Tensor* tensor) {
    CHECK(tensor->dims_size() == 4, "MSRA filler only dims 4.")
    float n;
    auto dims = tensor->dims();
    int fan_in = dims[1] * dims[2] * dims[3];
    int fan_out = dims[0] * dims[2] * dims[3];
    if (varianceType_ == VarianceType::FANIN) {
        n = fan_in;
    } else if (varianceType_ == VarianceType::FANOUT) {
        n = fan_out;
    } else {
        n = (fan_in + fan_out) / 2.0f;
    }

    float std = sqrtf(2.0f / n);
    if (have_seed_) {
        Distributions::normal(tensor, 0.0f, std, seed_);
    } else {
        Distributions::normal(tensor, 0.0f, std);
    }
}


} // namespace utils
} // namespace eutopia
