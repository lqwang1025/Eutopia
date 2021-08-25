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
#include "utils/filler.h"
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
            absl::string_view strip_line = line;
            absl::ConsumePrefix(&strip_line, " ");
            absl::ConsumeSuffix(&strip_line, " ");
            if (strip_line == "" || strip_line[0] == '#') continue;
            if (strip_line[0] == '[') {
                line = line.substr(1, line.size()-2);                
                if (param_parse_methods.count(line) != 0) {
                    (this->*param_parse_methods[line])(file, graph);
                } else {
                    EU_ERROR<<"Unsupport " <<strip_line<<" type."<<EU_ENDL;
                }
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
    op::InputParam input_param;
    std::vector<std::string> producers;
    core::ir::Node* node = graph->add_node();
    for (const auto& it : params) {
        std::vector<std::string> param = absl::StrSplit(it, absl::ByChar('='));
        if(param.size() == 1) continue;
        CHECK(param.size() > 1, "Wrong config file format.");
        if (param[0] == "name") {
            input_param.op_name = param[1];
        } else if (param[0] == "dims") {
            neb::CJsonObject c_json;
            if (!c_json.Parse(param[1].c_str())) {
                EU_ERROR<<"Invaild json format"<<EU_ENDL;
            }
            uint32_t batch, height, width, channels;
            c_json["shape"].Get("batch", batch);
            c_json["shape"].Get("height", height);
            c_json["shape"].Get("width", width);
            c_json["shape"].Get("channels", channels);
            input_param.input_dims = {batch, height, width, channels};
            node->set_input_shape(input_param.input_dims);
        } else if (param[0] == "preprocess") {
            neb::CJsonObject c_json;
            if (!c_json.Parse(param[1].c_str())) {
                EU_ERROR<<"Invaild json format"<<EU_ENDL;
            }
            c_json.Get("mean", input_param.mean);
            c_json.Get("std", input_param.std);
        } else if (param[0] == "producer") {
            std::vector<std::string> p_param= absl::StrSplit(param[1].substr(1, param[1].size()-2), absl::ByChar(','));
            for (auto it : p_param) {
                absl::string_view param_item = it;
                do {
                    absl::ConsumePrefix(&param_item, " ");
                    absl::ConsumeSuffix(&param_item, " ");
                } while (param_item[0] == ' ' || param_item[param_item.size()-1] == ' ');
                if (param_item != "") {
                    node->add_producer(std::string(param_item));
                }
            }
            if (node->get_producers().size() == 0) {
                input_param.first_op = true;
            }
        } else {
            EU_WARN<<"Need to add param " <<param[0]<<" to input node."<<EU_ENDL;
        }
    }
    node->setup(&input_param);
}

void CfgParser::init_conv2d_param(std::fstream& file, core::ir::Graph* graph) {
    std::vector<std::string> params = get_param(file);
    op::Convolution2DParam conv_param;
    std::vector<std::string> producers;
    core::ir::Node* node = graph->add_node();
    for (const auto& it : params) {
        std::vector<std::string> param = absl::StrSplit(it, absl::ByChar('='));
        if(param.size() == 1) continue;
        CHECK(param.size() > 1, "Wrong config file format.");
        if (param[0] == "name") {
            conv_param.op_name = param[1];
        } else if (param[0] == "op_param") {
            neb::CJsonObject c_json;
            if (!c_json.Parse(param[1].c_str())) {
                EU_ERROR<<"Invaild json format"<<EU_ENDL;
            }
            uint32_t oc, kh, kw, ic = 0;
            c_json.Get("filters", oc);
            uint32_t value = 0;
            CHECK(c_json["kernels"].GetArraySize() > 1, "Conv parameter kernels size must be greater than 1.");
            c_json["kernels"].Get(0, kh);
            c_json["kernels"].Get(1, kw);
            if (c_json["kernels"].GetArraySize() > 3) {
                c_json["kernels"].Get(1, ic);
            }
            conv_param.kernel_shape = {kh, kw, ic, oc};
            
            conv_param.strides.resize(2);
            CHECK(c_json["strides"].GetArraySize() == 2, "Conv parameter stride size must be 2.");
            c_json["strides"].Get(0, conv_param.strides[0]);
            c_json["strides"].Get(1, conv_param.strides[1]);
            conv_param.pad_type = c_json("padding");
            
            CHECK(c_json["pads"].GetArraySize() == 4, "Conv parameter pads size must be 4.");
            conv_param.pads.resize(4);
            c_json["pads"].Get(0, conv_param.pads[0]);
            c_json["pads"].Get(1, conv_param.pads[1]);
            c_json["pads"].Get(2, conv_param.pads[2]);
            c_json["pads"].Get(3, conv_param.pads[3]);

            CHECK(c_json["dilations"].GetArraySize() == 4, "Conv parameter dilations size must be 4.");
            conv_param.dilations.resize(4);
            c_json["dilations"].Get(0, conv_param.dilations[0]);
            c_json["dilations"].Get(1, conv_param.dilations[1]);
            c_json["dilations"].Get(2, conv_param.dilations[2]);
            c_json["dilations"].Get(3, conv_param.dilations[3]);
            // std::cout<<"debug:"<<c_json("activation")<<std::endl;
            c_json.Get("group", conv_param.group);
            
        } else if (param[0] == "weight_filler") {
            absl::string_view param_item = param[1];
            do {
                absl::ConsumePrefix(&param_item, " ");
                absl::ConsumeSuffix(&param_item, " ");
            } while (param_item[0] == ' ' || param_item[param_item.size()-1] == ' ');
            if (param_item == "xavier") {
                utils::XavierFiller* filler = new utils::XavierFiller;
                node->set_weight_filler(filler);
            }
        } else if (param[0] == "bias_filler") {
            absl::string_view param_item = param[1];
            do {
                absl::ConsumePrefix(&param_item, " ");
                absl::ConsumeSuffix(&param_item, " ");
            } while (param_item[0] == ' ' || param_item[param_item.size()-1] == ' ');
            if (param_item == "constant") {
                utils::ConstantFiller* filler = new utils::ConstantFiller(1.0);//TODO
                node->set_bias_filler(filler);
            }
        } else if (param[0] == "producer") {
            std::vector<std::string> p_param= absl::StrSplit(param[1].substr(1, param[1].size()-2), absl::ByChar(','));
            for (auto it : p_param) {
                absl::string_view param_item = it;
                do {
                    absl::ConsumePrefix(&param_item, " ");
                    absl::ConsumeSuffix(&param_item, " ");
                } while (param_item[0] == ' ' || param_item[param_item.size()-1] == ' ');
                if (param_item != "") {
                    node->add_producer(std::string(param_item));
                }
            }
            if (node->get_producers().size() == 0) {
                conv_param.first_op = true;
            }
        } else {
            EU_WARN<<"Need to add param " <<param[0]<<" to conv2d node."<<EU_ENDL;
        }
    }
    node->setup(&conv_param);
}

void CfgParser::init_pooling_param(std::fstream& file, core::ir::Graph* graph) {
    std::vector<std::string> params = get_param(file);
    op::PoolingParam pool_param;
    std::vector<std::string> producers;
    core::ir::Node* node = graph->add_node();
    for (const auto& it : params) {
        std::vector<std::string> param = absl::StrSplit(it, absl::ByChar('='));
        if(param.size() == 1) continue;
        CHECK(param.size() > 1, "Wrong config file format.");
        if (param[0] == "name") {
            pool_param.op_name = param[1];
        } else if (param[0] == "op_param") {
            neb::CJsonObject c_json;
            if (!c_json.Parse(param[1].c_str())) {
                EU_ERROR<<"Invaild json format"<<EU_ENDL;
            }
            
            CHECK(c_json["kernels"].GetArraySize() == 2, "Pool parameter kernels size must be equal 2.");
            pool_param.kernels.resize(2);
            c_json["kernels"].Get(0, pool_param.kernels[0]);
            c_json["kernels"].Get(1, pool_param.kernels[1]);
            
            pool_param.strides.resize(2);
            CHECK(c_json["strides"].GetArraySize() == 2, "Pool parameter stride size must be 2.");
            c_json["strides"].Get(0, pool_param.strides[0]);
            c_json["strides"].Get(1, pool_param.strides[1]);
            
            pool_param.pool_type = c_json("pool_type");
            
        } else if (param[0] == "producer") {
            std::vector<std::string> p_param= absl::StrSplit(param[1].substr(1, param[1].size()-2), absl::ByChar(','));
            for (auto it : p_param) {
                absl::string_view param_item = it;
                do {
                    absl::ConsumePrefix(&param_item, " ");
                    absl::ConsumeSuffix(&param_item, " ");
                } while (param_item[0] == ' ' || param_item[param_item.size()-1] == ' ');
                if (param_item != "") {
                    node->add_producer(std::string(param_item));
                }
            }
        } else {
            EU_WARN<<"Need to add param " <<param[0]<<" to pooling node."<<EU_ENDL;
        }
    }
    node->setup(&pool_param);
}

void CfgParser::init_fc_param(std::fstream& file, core::ir::Graph* graph) {
    std::vector<std::string> params = get_param(file);
    op::FullyConnectedParam fc_param;
    std::vector<std::string> producers;
    core::ir::Node* node = graph->add_node();
    for (const auto& it : params) {
        std::vector<std::string> param = absl::StrSplit(it, absl::ByChar('='));
        if(param.size() == 1) continue;
        CHECK(param.size() > 1, "Wrong config file format.");
        if (param[0] == "name") {
            fc_param.op_name = param[1];
        } else if (param[0] == "op_param") {
            neb::CJsonObject c_json;
            if (!c_json.Parse(param[1].c_str())) {
                EU_ERROR<<"Invaild json format"<<EU_ENDL;
            }
            c_json.Get("num_outputs", fc_param.num_outputs);
            
        } else if (param[0] == "producer") {
            std::vector<std::string> p_param= absl::StrSplit(param[1].substr(1, param[1].size()-2), absl::ByChar(','));
            for (auto it : p_param) {
                absl::string_view param_item = it;
                do {
                    absl::ConsumePrefix(&param_item, " ");
                    absl::ConsumeSuffix(&param_item, " ");
                } while (param_item[0] == ' ' || param_item[param_item.size()-1] == ' ');
                if (param_item != "") {
                    node->add_producer(std::string(param_item));
                }
            }
        } else if (param[0] == "weight_filler") {
            absl::string_view param_item = param[1];
            do {
                absl::ConsumePrefix(&param_item, " ");
                absl::ConsumeSuffix(&param_item, " ");
            } while (param_item[0] == ' ' || param_item[param_item.size()-1] == ' ');
            if (param_item == "xavier") {
                utils::XavierFiller* filler = new utils::XavierFiller;
                node->set_weight_filler(filler);
            }
        } else if (param[0] == "bias_filler") {
            absl::string_view param_item = param[1];
            do {
                absl::ConsumePrefix(&param_item, " ");
                absl::ConsumeSuffix(&param_item, " ");
            } while (param_item[0] == ' ' || param_item[param_item.size()-1] == ' ');
            if (param_item == "constant") {
                utils::ConstantFiller* filler = new utils::ConstantFiller(1.0);//TODO
                node->set_bias_filler(filler);
            }
        } else {
            EU_WARN<<"Need to add param " <<param[0]<<" to fc node."<<EU_ENDL;
        }
    }
    node->setup(&fc_param);
}

} // namespace io
} // namespace eutopia
