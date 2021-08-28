# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# License); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# (C) COPYRIGHT Daniel Wang Limited.
# File       : CMakeLists.txt
# Authors    : Daniel Wang
# Create Time: 2021-08-08:13:30:41
# Email      : wangliquan21@qq.com
# Description:

file(GLOB_RECURSE MAIN_SRCS *.cpp)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/third_party")
set(PROTOBUF_DIR "${THIRD_PARTY_DIR}/protobuf")
set(ABSEIL_DIR "${THIRD_PARTY_DIR}/abseil")
message("${PROTOBUF_DIR}")

link_directories(${PROTOBUF_DIR}/lib)
link_directories(${ABSEIL_DIR}/lib)
link_directories(${CMAKE_BINARY_DIR})

add_executable(eutopia-train-tools ${MAIN_SRCS})

target_link_libraries(eutopia-train-tools  libeutopia.a libabsl.a libprotobuf.a libprotobuf-lite.a libprotoc.a)
