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
# Create Time: 2021-08-08:08:26:19
# Email      : wangliquan21@qq.com
# Description:

# Check CMake version
CMAKE_MINIMUM_REQUIRED (VERSION 3.13 FATAL_ERROR)

# Disable inplace builds to prevent source tree corruption.
IF (" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  MESSAGE (FATAL_ERROR "Eutopia Fatal: Building inplace are not allowed. You should create a separate directory for Building.")
ENDIF ()

# set cmake_install_prefix path
IF (NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Eutopia Installation Directory")
ENDIF()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# Enable the languages which in use
ENABLE_LANGUAGE (C CXX)

IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
ENDIF()

# Project name
PROJECT (eutopia)

SET(CMAKE_CXX_STANDARD 11)
SET(CXX_STANDARD_REQUIRED ON)

exec_program(protoc ARGS "--version 2>&1 | sed -n '/libprotoc/p' | awk -F ' ' '{print $2}'" OUTPUT_VARIABLE PROTO_VERSION RETURN_VALUE CMD_RETURN)
if(${PROTO_VERSION} VERSION_LESS "3.6.0")
  message(FATAL_ERROR "protoc version must be greater than 3.6.0.")
endif()

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party")
set(PROTOBUF_DIR "${THIRD_PARTY_DIR}/protobuf")
set(CJSON_DIR "${THIRD_PARTY_DIR}/CJsonObject")
set(ABSEIL_DIR "${THIRD_PARTY_DIR}/abseil")

file(GLOB_RECURSE PROTO_FILE_LIST_TMP ${CMAKE_CURRENT_LIST_DIR}/core/framework/*.proto)
foreach(FILE_TMP ${PROTO_FILE_LIST_TMP})
  file(RELATIVE_PATH FILE_RES ${PROJECT_SOURCE_DIR} ${FILE_TMP})
  execute_process(COMMAND protoc --proto_path=./ --cpp_out=./ ${FILE_RES}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endforeach()

EXEC_PROGRAM(git  ARGS  "log --pretty=format:%h --max-count=1 | awk '{print $1}'" OUTPUT_VARIABLE CMD_OUT RETURN_VALUE CMD_RET)
STRING(TIMESTAMP BUILD_TIME "%Y%m%d-%H%M")
SET(BUILD_INFO "${BUILD_TIME}")
ADD_DEFINITIONS(-D__BUILD_INFO__=${BUILD_INFO})
ADD_DEFINITIONS(-D__GIT_VER__=${CMD_OUT})
message(STATUS "BUILD_INFO: " ${BUILD_INFO})
message(STATUS "GIT_VER: " ${CMD_OUT})

# building flags
ADD_DEFINITIONS(-Wl,-Bsymbolic)
ADD_DEFINITIONS(-Wl,-Bsymbolic-functions)
ADD_DEFINITIONS(-Wno-unused-command-line-argument)
ADD_DEFINITIONS(-funroll-loops)
ADD_DEFINITIONS(-Wno-deprecated-register)
ADD_DEFINITIONS(-Wno-overloaded-virtual)
ADD_DEFINITIONS(-fPIC)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

OPTION(DEBUGEUTOPIA "This option is for open debug mode" ON)

if(DEBUGEUTOPIA)
  ADD_DEFINITIONS(-DDEBUGEUTOPIA)
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
endif()

# add all header file folder
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROTOBUF_DIR}/include)
include_directories(${CJSON_DIR})
include_directories(${ABSEIL_DIR}/include)

link_directories(${PROTOBUF_DIR}/lib)
link_directories(${ABSEIL_DIR}/lib)

#add source file
file(GLOB CORE_SRC       "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"   "${CMAKE_CURRENT_SOURCE_DIR}/core/ir/*.cpp")
file(GLOB CORE_FRAMEWOEK_SRC       "${CMAKE_CURRENT_SOURCE_DIR}/core/framework/*.cc" "${CMAKE_CURRENT_SOURCE_DIR}/core/framework/*.cpp")
file(GLOB_RECURSE OP_SRC       "${CMAKE_CURRENT_SOURCE_DIR}/op/*.cpp")
file(GLOB_RECURSE IO_SRC       "${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp")
file(GLOB THIRD_SRC       "${CMAKE_CURRENT_SOURCE_DIR}/third_party/CJsonObject/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/CJsonObject/*.c")

list(APPEND EUPOTIA_SRC
  ${CORE_SRC} ${CORE_FRAMEWOEK_SRC}
  ${OP_SRC} ${THIRD_SRC} ${IO_SRC})

# add header file
file(GLOB CORE_HEAD       "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"  "${CMAKE_CURRENT_SOURCE_DIR}/core/ir/*.h")

# Report summary
INCLUDE ("${CMAKE_CURRENT_SOURCE_DIR}/cmake/summary.cmake")

#add exec
ADD_LIBRARY (${CMAKE_PROJECT_NAME}-static STATIC)
ADD_LIBRARY (${CMAKE_PROJECT_NAME}        SHARED)

# add source files to each target
TARGET_SOURCES (${CMAKE_PROJECT_NAME}-static PRIVATE ${EUPOTIA_SRC})
TARGET_SOURCES (${CMAKE_PROJECT_NAME}        PRIVATE ${EUPOTIA_SRC})

target_link_libraries(${CMAKE_PROJECT_NAME}-static)
target_link_libraries(${CMAKE_PROJECT_NAME})

#set version
set_target_properties(${CMAKE_PROJECT_NAME}-static PROPERTIES VERSION 0.0.0 SOVERSION 0)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES VERSION 0.0.0 SOVERSION 0)

set_target_properties(${CMAKE_PROJECT_NAME}-static PROPERTIES OUTPUT_NAME "eutopia")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME "eutopia")

# # install tengine libraries
INSTALL (TARGETS ${CMAKE_PROJECT_NAME}        LIBRARY DESTINATION lib)
INSTALL (TARGETS ${CMAKE_PROJECT_NAME}-static ARCHIVE DESTINATION lib)

# install header files
INSTALL (FILES ${OP_HEAD} ${CORE_HEAD} DESTINATION include/eutopia/)

add_subdirectory(tools)

install(TARGETS eutopia-train-tools RUNTIME DESTINATION bin)
