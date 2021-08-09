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
 * File       : data_type.cpp
 * Authors    : Daniel Wang
 * Create Time: 2021-08-08:13:15:20
 * Email      : wangliquan21@qq.com
 * Description:
 */

#include <string>
#include <map>

#include "core/data_type.h"

namespace eutopia {
namespace core {

struct DataTypeInfo {
    std::string data_type_str;
    uint8_t byte_size;
};

static const std::map<DataType, DataTypeInfo> data_type_info_map = {
    {DataType::EUTOPIA_DT_UNKNOWN, {"EUTOPIA_DT_UNKNOWN", 0}},
    {DataType::EUTOPIA_DT_UINT8, {"EUTOPIA_DT_UINT8", 1}},
    {DataType::EUTOPIA_DT_INT8, {"EUTOPIA_DT_INT8", 1}},
    {DataType::EUTOPIA_DT_INT16, {"EUTOPIA_DT_INT16", 2}},
    {DataType::EUTOPIA_DT_UINT16, {"EUTOPIA_DT_UINT16", 2}},
    {DataType::EUTOPIA_DT_INT32, {"EUTOPIA_DT_INT32", 4}},
    {DataType::EUTOPIA_DT_UINT32, {"EUTOPIA_DT_UINT32", 4}},
    {DataType::EUTOPIA_DT_FP32, {"EUTOPIA_DT_FP32", 4}},
};

std::string data_type_to_string(DataType data_type) {
    if (data_type_info_map.count(data_type) != 0) {
        return data_type_info_map.at(data_type).data_type_str;
    } else {
        return "EUTOPIA_DT_MAX";
    }
}

DataType string_to_data_type(std::string& data_type_str) {
    if (data_type_str == "EUTOPIA_DT_UNKNOWN") {
        return DataType::EUTOPIA_DT_UNKNOWN;
    } else if (data_type_str == "EUTOPIA_DT_UINT8") {
        return DataType::EUTOPIA_DT_UINT8;
    } else if (data_type_str == "EUTOPIA_DT_INT8") {
        return DataType::EUTOPIA_DT_INT16;
    } else if (data_type_str == "EUTOPIA_DT_INT16") {
        return DataType::EUTOPIA_DT_INT16;
    } else if (data_type_str == "EUTOPIA_DT_INT32") {
        return DataType::EUTOPIA_DT_INT32;
    } else if (data_type_str == "EUTOPIA_DT_UINT32") {
        return DataType::EUTOPIA_DT_UINT32;
    } else if (data_type_str == "EUTOPIA_DT_FP32") {
        return DataType::EUTOPIA_DT_FP32;
    } else {
        return DataType::EUTOPIA_DT_MAX;
    }
}

uint8_t get_data_type_byte(DataType data_type) {
    if (data_type_info_map.count(data_type) != 0) {
        return data_type_info_map.at(data_type).byte_size;
    } else {
        return 0;
    }
}

uint8_t get_data_type_byte(std::string& data_type_str) {
    DataType data_type = string_to_data_type(data_type_str);
    return get_data_type_byte(data_type);
}

} // namespace core
} // namespace eutopia
