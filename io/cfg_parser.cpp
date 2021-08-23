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
 * File       : cfg_parser.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-22:07:06:18
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/strings/string_view.h"
#include "core/ir/graph.h"
#include "io/cfg_parser.h"
#include "CJsonObject.hpp"

namespace eutopia {
namespace io {

core::ir::Graph* CfgParser::operator() (const char* file_name) {
    core::ir::Graph* graph = new core::ir::Graph;
    std::fstream file;
    file.open(file_name, std::fstream::in);
    if (file.is_open()) {
        std::string line;
        while(std::getline(file, line)) {
            absl::string_view strip_line = line.c_str();
            absl::ConsumePrefix(&strip_line, " ");
            absl::ConsumeSuffix(&strip_line, " ");
            if (strip_line == "") continue;
            if (strip_line[0] == '#')  continue;
            if (strip_line[0] == '[') {
                absl::ConsumePrefix(&strip_line, "[");
                absl::ConsumeSuffix(&strip_line, "]");
                std::cout<<"line:"<<line<<std::endl;
                // if (param_parse_methods.count(strip_line.c_str()) != 0) {
                //     (this->*param_parse_methods[strip_line.c_str()])(file, graph);
                // } else {
                //     EU_ERROR<<"Unsupport " <<strip_line<<" type."<<EU_ENDL;
                // }
            }
        }
    } else {
        EU_ERROR<<"Open" <<file_name<<" failed."<<EU_ENDL;
    }
    file.close();
    return graph;
}

std::vector<std::string> CfgParser::get_param(std::fstream& file) {
    std::string line, param;
    while (std::getline(file, line)) {
        if (line[0] == '[') {
            file.seekg(-line.size()-1, std::ios::cur);
            break;
        } else if (line[0] == '#' || line == "") {
            continue;
        } else {
            param += line;
        }
    }
    std::vector<std::string> params = absl::StrSplit(param, absl::ByChar(';'));
    return params;
}

void CfgParser::init_graph(std::fstream& file, core::ir::Graph* graph) {
    std::vector<std::string> params = get_param(file);
    for (auto& it : params) {
        std::vector<std::string> param = absl::StrSplit(it, absl::ByChar('='));
        if(param.size() == 1) continue;
        CHECK(param.size() > 1, "Wrong config file format.");
        if (param[0] == "name") {
            graph->set_name(param[1]);
        } else if (param[0] == "training") {
            if (param[1] == "true") {
                graph->set_is_trainning(true);
            } else {
                graph->set_is_trainning(false);
            }
        } else {
            EU_WARN<<"Need to add param " <<param[0]<<" to graph."<<EU_ENDL;
        }
    }
}

void CfgParser::init_base_param(std::fstream& file, core::ir::Graph* graph) {
    std::cout<<__func__<<std::endl;
}

void CfgParser::init_input_param(std::fstream& file, core::ir::Graph* graph) {
    std::vector<std::string> params = get_param(file);
    for (auto& it : params) {
        std::cout<<it<<std::endl;
    }
}

void CfgParser::init_conv2d_param(std::fstream& file, core::ir::Graph* graph) {
    std::vector<std::string> params = get_param(file);
    for (auto& it : params) {
        std::cout<<it<<std::endl;
    }
}

void CfgParser::init_pooling_param(std::fstream& file, core::ir::Graph* graph) {
    std::cout<<__func__<<std::endl;
}

void CfgParser::init_fc_param(std::fstream& file, core::ir::Graph* graph) {
    std::cout<<__func__<<std::endl;
}

} // namespace io
} // namespace eutopia
