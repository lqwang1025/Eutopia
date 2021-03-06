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
 * File       : distributions.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-25:15:21:31
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __DISTRIBUTIONS_H__
#define __DISTRIBUTIONS_H__

#include <random>

namespace eutopia {

namespace core {
namespace ir {
class Tensor;
}
}

namespace utils {

struct Distributions {
    static void uniform(core::ir::Tensor* tensor, float min, float max,
                        int seed = std::random_device()());

    static void normal(core::ir::Tensor* tensor, float mean, float std,
                       int seed = std::random_device()());
    
    static void truncated_normal(core::ir::Tensor* tensor, float mean, float std,
                                 int seed = std::random_device()());
};

} // namespace utils
} // namespace eutopia

#endif /* __DISTRIBUTIONS_H__ */

