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
# Create Time: 2021-08-12:07:58:16
# Email      : wangliquan21@qq.com
# Description:
# set target name

IF (NOT DEFINED EUTOPIA_OP_NAME)
  SET (EUTOPIA_OP_NAME ${CMAKE_PROJECT_NAME}-op CACHE INTERNAL "" FORCE)
ENDIF()

UNSET (EUTOPIA_OP_SOURCE_FILES)
UNSET (EUTOPIA_OP_HEADER_PATH)

UNSET (EUTOPIA_CPP_FILE)
UNSET (EUTOPIA_CC_FILE)
UNSET (EUTOPIA_C_FILE)

file(GLOB_RECURSE EUTOPIA_CPP_FILE *.cpp)
file(GLOB_RECURSE EUTOPIA_CC_FILE *.cc)
file(GLOB_RECURSE EUTOPIA_C_FILE *.c)

LIST (APPEND EUTOPIA_OP_SOURCE_FILES       ${EUTOPIA_CPP_FILE})
LIST (APPEND EUTOPIA_OP_SOURCE_FILES       ${EUTOPIA_C_FILE})
LIST (APPEND EUTOPIA_OP_SOURCE_FILES       ${EUTOPIA_CC_FILE})

# add compiler options
LIST (APPEND EUTOPIA_COMPILER_OPTIONS   ${EUTOPIA_COMMON_OPTIONS})

# add linker options
LIST (APPEND EUTOPIA_LINKER_OPTIONS     ${EUTOPIA_COMMON_LINKER_OPTIONS})

# add static and shared target
ADD_LIBRARY (${EUTOPIA_OP_NAME}-static STATIC)
ADD_LIBRARY (${EUTOPIA_OP_NAME}        SHARED)

# add source files to each target
TARGET_SOURCES (${EUTOPIA_OP_NAME}-static PRIVATE ${EUTOPIA_OP_SOURCE_FILES})
TARGET_SOURCES (${EUTOPIA_OP_NAME}        PRIVATE ${EUTOPIA_OP_SOURCE_FILES})

# add options
TARGET_COMPILE_OPTIONS (${EUTOPIA_OP_NAME}-static PUBLIC $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${EUTOPIA_COMPILER_OPTIONS}>)
TARGET_COMPILE_OPTIONS (${EUTOPIA_OP_NAME}        PUBLIC $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${EUTOPIA_COMPILER_OPTIONS}>)

#set .a version
set_target_properties(${EUTOPIA_OP_NAME}-static PROPERTIES VERSION 0.0.0 SOVERSION 0)

#set .so version
set_target_properties(${EUTOPIA_OP_NAME} PROPERTIES VERSION 0.0.0 SOVERSION 0)

set_target_properties(${EUTOPIA_OP_NAME}-static PROPERTIES OUTPUT_NAME "eutopia-op")
set_target_properties(${EUTOPIA_OP_NAME} PROPERTIES OUTPUT_NAME "eutopia-op")
