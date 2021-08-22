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
 * File       : cfg_parser.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-22:07:04:35
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __CFG_PARSER_H__
#define __CFG_PARSER_H__

#include <vector>
#include <fstream>

#include "op/ops_param.h"

namespace eutopia {
namespace io {

struct CfgParser {
    std::vector<op::BaseParam*> operator() (const char* file_name);
};

struct BaseParamParser {
    op::BaseParam* operator() (std::fstream& file);
};

struct Conv2dParamParser {
    op::Convolution2DParam* operator() (std::fstream& file);
};

struct PoolingParamParser {
    op::PoolingParam* operator() (std::fstream& file);
};


} // namespace io
} // namespace eutopia

#endif /* __CFG_PARSER_H__ */

