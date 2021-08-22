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
#include "io/cfg_parser.h"
#include "CJsonObject.hpp"

namespace eutopia {
namespace io {

std::vector<op::BaseParam*> CfgParser::operator() (const char* file_name) {
    std::vector<op::BaseParam*> params;
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
                std::cout<<strip_line<<std::endl;
                // std::string ss =
                char c;
                std::string name("");
                do {
                    file.get(c);
                    name += c;
                } while (c != ';');
                std::cout<<name<<std::endl;
            }
        }
    } else {
        EU_ERROR<<"Open" <<file_name<<" failed."<<EU_ENDL;
    }
    file.close();
    return params;
}

op::BaseParam* BaseParamParser::operator() (std::fstream& file) {
    
}

op::Convolution2DParam* Conv2dParamParser::operator() (std::fstream& file) {
    
}

op::PoolingParam* PoolingParamParser::operator() (std::fstream& file) {
    
}

} // namespace io
} // namespace eutopia
