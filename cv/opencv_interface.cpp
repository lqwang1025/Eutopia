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
 * File       : opencv_interface.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-31:07:35:55
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifdef WITH_OPENCV 1

#include <iostream>
#include <cstddef>

#include "cv/opencv_interface.h"
#include "core/ir/tensor.h"
#include "core/data_type.h"
#include "core/logging.h"

namespace eutopia {
namespace cv {

core::ir::Tensor* mats_to_tensor(std::vector<::cv::Mat>& mats) {
    if(mats.size() == 0) return nullptr;
    std::vector<uint32_t> tensor_shape = {mats.size(), mats[0].channels(), mats[0].rows, mats[0].cols};
    core::ir::Tensor* tensor = new core::ir::Tensor(tensor_shape, core::DataType::EUTOPIA_DT_UINT8);
    // for (int i = 0; i < (int)mats.size(); ++i) {
        
    // }
    return tensor;
}

::cv::Mat* mat_to_tensor(const core::ir::Tensor& tensor) {
    
}

} // namespace cv
} // namespace eutopia

#endif
