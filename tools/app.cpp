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
 * File       : app.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-08:13:30:59
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>

#include "io/parser.h"
#include "core/ir/graph.h"
#include "core/ir/tensor.h"
#include "tools/cmdline.h"
#include "cv/opencv_interface.h"

int main(int argc, char** argv) {
    eutopia::io::Parser p;
    cmdline::parser a;
    a.set_program_name("eutopia-train-tools");
    a.parse_check(argc, argv);
    eutopia::core::ir::Graph* graph = p.run("/home/parallels/project/Eutopia/tools/cfg/alexnet.cfg");
    ::cv::Mat pic = ::cv::imread("./cat.jpg");
    std::vector<::cv::Mat> mats = {pic};
    eutopia::cv::mats_to_tensor(mats);
    cv::imshow("pcv", pic);
    cv::waitKey(0);
    eutopia::core::ir::Tensor* tensor = new eutopia::core::ir::Tensor({20, 1, 28, 28}, eutopia::core::DataType::EUTOPIA_DT_FP32);
    graph->warm_up();
    graph->forward(tensor);
    graph->to_proto();
    delete graph;
    delete tensor;
    return 0;
}
