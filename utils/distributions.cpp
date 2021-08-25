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
 * File       : distributions.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-25:15:21:31
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <random>
#include <cstring>

#include "core/ir/tensor.h"
#include "core/ir/chunk.h"
#include "core/data_type.h"
#include "core/logging.h"
#include "utils/distributions.h"

namespace eutopia {
namespace utils {

template<typename T>
static void _data_fill(core::ir::Tensor* tensor, T distribution, std::mt19937 mt) {
    core::ir::Chunk* chunk = tensor->chunk();
    uint32_t byte_size = chunk->get_byte_size();
    CHECK(byte_size>0, "Wrong byte size number.");
    float* data_ptr = (float*)chunk->get_mutable_data_ptr();
    memset(data_ptr, 0, byte_size);
    for (int i = 0; i < (byte_size>>2); ++i) {
        data_ptr[i] = distribution(mt);
    }
}

static float _get_truncated_data(std::normal_distribution<float>& distribution, std::mt19937& mt) {
    auto mean = distribution.mean();
    auto std = distribution.stddev();
    float min = mean - 2 * std;
    float max = mean + 2 * std;
    float value;
    do {
        value = distribution(mt);
    } while (value < min || value > max);
    return value;
}


template<typename T>
static void _truncated_data_fill(core::ir::Tensor* tensor, T distribution, std::mt19937 mt) {
    core::ir::Chunk* chunk = tensor->chunk();
    uint32_t byte_size = chunk->get_byte_size();
    CHECK(byte_size>0, "Wrong byte size number.");
    float* data_ptr = (float*)chunk->get_mutable_data_ptr();
    memset(data_ptr, 0, byte_size);
    for (int i = 0; i < (byte_size>>2); ++i) {
        data_ptr[i] = _get_truncated_data(distribution, mt);
    }
}

void Distributions::uniform(core::ir::Tensor* tensor, float min, float max, int seed) {
    CHECK(tensor->get_data_type() == core::DataType::EUTOPIA_DT_FP32, "Filler now only support float weight.");
    CHECK(min <= max, "Are you kidding me?");
    std::mt19937 mt(seed);
    std::uniform_real_distribution<float> uniform(min, max);
    _data_fill(tensor, uniform, mt);
    
}

void Distributions::normal(core::ir::Tensor* tensor, float mean, float std, int seed) {
    CHECK(tensor->get_data_type() == core::DataType::EUTOPIA_DT_FP32, "Filler now only support float weight.");
    std::mt19937 mt(seed);
    std::normal_distribution<float> normal(mean, std);
    _data_fill(tensor, normal, mt);
}
    
void Distributions::truncated_normal(core::ir::Tensor* tensor, float mean, float std, int seed) {
    CHECK(tensor->get_data_type() == core::DataType::EUTOPIA_DT_FP32, "Filler now only support float weight.");
    std::mt19937 mt(seed);
    std::normal_distribution<float> truncated_normal(mean, std);
    _truncated_data_fill(tensor, truncated_normal, mt);
}

} // namespace utils
} // namespace eutopia

