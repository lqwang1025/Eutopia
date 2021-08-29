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

#include <map>
#include <vector>
#include <fstream>
#include <string>

#include "op/ops_param.h"
#include "core/ir/graph.h"
#include "CJsonObject.hpp"

namespace eutopia {
namespace utils {
class Filler;
}
namespace core {
namespace ir {
class Node;
}
}
namespace io {

struct CfgParser {
    core::ir::Graph* operator() (const char* file_name);
    std::vector<std::string> get_param(std::fstream& file);
    typedef void (CfgParser::*ParamFunc)(std::fstream& file, core::ir::Graph* graph);
    void init_graph(std::fstream& file, core::ir::Graph* graph);
    void init_base_param(std::fstream& file, core::ir::Graph* graph);
    void init_input_param(std::fstream& file, core::ir::Graph* graph);
    void init_fc_param(std::fstream& file, core::ir::Graph* graph);
    void init_conv2d_param(std::fstream& file, core::ir::Graph* graph);
    void init_pooling_param(std::fstream& file, core::ir::Graph* graph);
    std::map<std::string, ParamFunc> param_parse_methods {
        {"Graph", &CfgParser::init_graph},
        {INPUT, &CfgParser::init_input_param},
        {FULLYCONNECTED, &CfgParser::init_fc_param},
        {POOLING, &CfgParser::init_pooling_param},
        {CONVOLUTION2D, &CfgParser::init_conv2d_param},
    };
private:
    utils::Filler* _init_filler_type_(const neb::CJsonObject& c_json);
};

} // namespace io
} // namespace eutopia

#endif /* __CFG_PARSER_H__ */

