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
 * File       : main.cc
 * Authors    : Daniel Wang
 * Create Time: 2021-08-08:13:30:59
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <iostream>
#include <cstdlib>

#include "core/data_type.h"
#include "core/ir/tensor.h"
#include "core/ir/node.h"
#include "core/ir/graph.h"
#include "core/graph_builder.h"
#include "op/ops_param/convolution2D_param.h"
#include "CJsonObject.hpp"

#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <cstdio>

using namespace std;

int main(int argc, char** argv) {
    // eutopia::core::ir::Tensor tensor;
    // std::vector<uint32_t> dims = {224, 224, 3};
    // tensor.set_data(dims, eutopia::core::DataType::EUTOPIA_DT_UINT8);
    // const uint8_t& s = tensor.data<uint8_t>({112, 112, 2});
    // std::cout<<(int)s<<std::endl;
    // tensor.mutable_data<uint8_t>({112, 112, 2}) = 255;
    eutopia::core::ir::Graph graph;
    struct eutopia::op::Convolution2DParam conv_param;
    conv_param.pad_type = "SAME";
    eutopia::core::ir::Node* node = graph.add_node(&conv_param);
    std::vector<int> s;
    s.push_back(10);
    s.push_back(8);
    std::cout<<"debug:"<<s[0]<<s[1]<<std::endl;
    
    int iValue;
    double fTimeout;
    std::string strValue;
    neb::CJsonObject oJson("{\"refresh_interval\":60,"
                        "\"test_float\":[18.0, 10.0, 5.0],"
                        "\"timeout\":12.5,"
                        "\"dynamic_loading\":["
                            "{"
                                "\"so_path\":\"plugins/User.so\", \"load\":false, \"version\":1,"
                                "\"cmd\":["
                                     "{\"cmd\":2001, \"class\":\"neb::CmdUserLogin\"},"
                                     "{\"cmd\":2003, \"class\":\"neb::CmdUserLogout\"}"
                                "],"
                                "\"module\":["
                                     "{\"path\":\"im/user/login\", \"class\":\"neb::ModuleLogin\"},"
                                     "{\"path\":\"im/user/logout\", \"class\":\"neb::ModuleLogout\"}"
                                "]"
                             "},"
                             "{"
                             "\"so_path\":\"plugins/ChatMsg.so\", \"load\":false, \"version\":1,"
                                 "\"cmd\":["
                                      "{\"cmd\":2001, \"class\":\"neb::CmdChat\"}"
                                 "],"
                             "\"module\":[]"
                             "}"
                        "]"
                    "}");
     std::cout << oJson.ToString() << std::endl;
     std::cout << "-------------------------------------------------------------------" << std::endl;
     std::cout << oJson["dynamic_loading"][0]["cmd"][1]("class") << std::endl;
     oJson["dynamic_loading"][0]["cmd"][0].Get("cmd", iValue);
     std::cout << "iValue = " << iValue << std::endl;
     oJson["dynamic_loading"][0]["cmd"][0].Replace("cmd", -2001);
     oJson["dynamic_loading"][0]["cmd"][0].Get("cmd", iValue);
     std::cout << "iValue = " << iValue << std::endl;
     oJson.Get("timeout", fTimeout);
     std::cout << "fTimeout = " << fTimeout << std::endl;
     oJson["dynamic_loading"][0]["module"][0].Get("path", strValue);
     std::cout << "strValue = " << strValue << std::endl;
     std::cout << "-------------------------------------------------------------------" << std::endl;
     oJson.AddEmptySubObject("depend");
     oJson["depend"].Add("nebula", "https://github.com/Bwar/Nebula");
     oJson["depend"].AddEmptySubArray("bootstrap");
     oJson["depend"]["bootstrap"].Add("BEACON");
     oJson["depend"]["bootstrap"].Add("LOGIC");
     oJson["depend"]["bootstrap"].Add("LOGGER");
     oJson["depend"]["bootstrap"].Add("INTERFACE");
     oJson["depend"]["bootstrap"].Add("ACCESS");
     std::cout << oJson.ToString() << std::endl;
     std::cout << "-------------------------------------------------------------------" << std::endl;
     std::cout << oJson.ToFormattedString() << std::endl;

     std::cout << "-------------------------------------------------------------------" << std::endl;
     neb::CJsonObject oCopyJson = oJson;
     if (oCopyJson == oJson)
     {
         std::cout << "json equal" << std::endl;
     }
     oCopyJson["depend"]["bootstrap"].Delete(1);
     oCopyJson["depend"].Replace("nebula", "https://github.com/Bwar/CJsonObject");
     std::cout << oCopyJson.ToString() << std::endl;
     std::cout << "-------------------------key traverse------------------------------" << std::endl;
     std::string strTraversing;
     while(oJson["dynamic_loading"][0].GetKey(strTraversing))
     {
         std::cout << "traversing:  " << strTraversing << std::endl;
     }
     std::cout << "---------------add a new key, then key traverse---------------------" << std::endl;
     oJson["dynamic_loading"][0].Add("new_key", "new_value");
     while(oJson["dynamic_loading"][0].GetKey(strTraversing))
     {
         std::cout << "traversing:  " << strTraversing << std::endl;
     }

     std::cout << oJson["test_float"].GetArraySize() << std::endl;
     float fTestValue = 0.0;
     for (int i = 0; i < oJson["test_float"].GetArraySize(); ++i)
     {
         oJson["test_float"].Get(i, fTestValue);
         std::cout << fTestValue << std::endl;
     }
     oJson.AddNull("null_value");
     std::cout << oJson.IsNull("test_float") << "\t" << oJson.IsNull("null_value") << std::endl;
     oJson["test_float"].AddNull();
     std::cout << oJson.ToString() << std::endl;
    return 0;
}
