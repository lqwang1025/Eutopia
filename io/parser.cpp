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
 * File       : parser.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-20:10:39:24
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "io/parser.h"
#include "core/logging.h"
#include "io/cfg_parser.h"
#include "absl/strings/str_split.h"

namespace eutopia {
namespace io {

Parser::Parser() {
    
}

Parser::~Parser() {
    
}

core::ir::Graph* Parser::run(const char* file_name) {
    std::vector<std::string> spilt_names = absl::StrSplit(file_name, absl::ByString("."));
    CHECK(spilt_names.size()!=0, "Get file name failed.");
    if (spilt_names[spilt_names.size()-1] == "cfg") {
        CfgParser cfg_parser;
        return cfg_parser(file_name);
    } else {
        EU_ERROR<<"Todo: support more cfg format.\n"<<EU_ENDL;
    }
}

} // namespace io
} // namespace eutopia
