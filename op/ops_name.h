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
 * File       : ops_name.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-12:07:49:42
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __OPS_NAME_H__
#define __OPS_NAME_H__

#define MUL "Mul"
#define POOL "Pool"
#define CONCAT "Concat"
#define DROPOUT "DropOut"
#define BATCHNORM "BatchNorm"
#define CONVOLUTION2D "Convolution2D"
#define FULLYCONNECTED "FullyConnected"
#define GLOBALAVGPOOL "GlobalAVGPool"
#define DEPTHWISECONVOLUTION2D "DepthWiseConvolution2D"

namespace eutopia {
namespace op {

static const std::set<std::string> EUTOPIA_SUPPORT_OPS = {
    MUL,
    CONCAT,
    POOL,
    DROPOUT,
    BATCHNORM,
    CONVOLUTION2D,
    FULLYCONNECTED,
    GLOBALAVGPOOL,
    DEPTHWISECONVOLUTION2D,
};

} // namespace op
} // namespace eutopia

#endif /* __OPS_NAME_H__ */

