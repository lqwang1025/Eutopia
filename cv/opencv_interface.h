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
 * File       : opencv_interface.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-31:07:35:49
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __OPENCV_INTERFACE_H__
#define __OPENCV_INTERFACE_H__

#ifdef WITH_OPENCV 1
#include <opencv2/core/core.hpp>

namespace eutopia {
namespace core {
namespace ir  {
class Tensor;
}
}
namespace cv {

core::ir::Tensor* mat_to_tensor(const ::cv::Mat& mat);

::cv::Mat* mat_to_tensor(const core::ir::Tensor& tensor);

} // namespace cv
} // namespace eutopia

#endif

#endif /* __OPENCV_INTERFACE_H__ */

