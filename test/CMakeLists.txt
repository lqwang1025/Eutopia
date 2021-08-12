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

UNSET (EUTOPIA_TEST_SOURCE_FILES)
UNSET (EUTOPIA_CPP_FILE)
UNSET (EUTOPIA_CC_FILE)
UNSET (EUTOPIA_C_FILE)

file(GLOB_RECURSE EUTOPIA_CPP_FILE *.cpp)
file(GLOB_RECURSE EUTOPIA_CC_FILE *.cc)
file(GLOB_RECURSE EUTOPIA_C_FILE *.c)

LIST (APPEND EUTOPIA_TEST_SOURCE_FILES       ${EUTOPIA_CPP_FILE})
LIST (APPEND EUTOPIA_TEST_SOURCE_FILES       ${EUTOPIA_C_FILE})
LIST (APPEND EUTOPIA_TEST_SOURCE_FILES       ${EUTOPIA_CC_FILE})

link_directories(${CMAKE_BINARY_DIR}/core)

add_executable(test_bin ${EUTOPIA_TEST_SOURCE_FILES})

target_link_libraries(test_bin eutopia-core)
