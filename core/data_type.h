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
 * File       : data_type.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-08:12:05:36
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include <stdint.h>
#include <string>

namespace eutopia {
namespace core {

enum class DataType : uint8_t {
  EUTOPIA_DT_UNKNOWN = 0x00,
  EUTOPIA_DT_UINT8 = 0x01,
  EUTOPIA_DT_INT8 = 0x02,
  EUTOPIA_DT_INT16 = 0x03,
  EUTOPIA_DT_UINT16 = 0x04,
  EUTOPIA_DT_INT32 = 0x05,
  EUTOPIA_DT_UINT32 = 0x06,
  EUTOPIA_DT_FP32 = 0x07,
  EUTOPIA_DT_MAX = 0xFF
};

std::string data_type_to_string(DataType data_type);

DataType string_to_data_type(std::string &data_type_str);

uint8_t get_data_type_byte(DataType data_type);

uint8_t get_data_type_byte(std::string &data_type_str);

} // namespace core
} // namespace eutopia
#endif /* __DATA_TYPE_H__ */
