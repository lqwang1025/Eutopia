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

#add source file
file(GLOB_RECURSE CORE_SRC       "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"          "${CMAKE_CURRENT_SOURCE_DIR}/core/ir/*.cpp")
file(GLOB_RECURSE OP_SRC       "${CMAKE_CURRENT_SOURCE_DIR}/op/*.cpp"          "${CMAKE_CURRENT_SOURCE_DIR}/op/*.cpp")

# add header file
file(GLOB_RECURSE CORE_HEAD       "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"          "${CMAKE_CURRENT_SOURCE_DIR}/core/ir/*.h")
file(GLOB_RECURSE OP_HEAD       "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"          "${CMAKE_CURRENT_SOURCE_DIR}/core/ir/*.h")

# Report summary
INCLUDE ("${CMAKE_CURRENT_SOURCE_DIR}/cmake/summary.cmake")

#add exec
ADD_LIBRARY (${CMAKE_PROJECT_NAME}-static STATIC)
ADD_LIBRARY (${CMAKE_PROJECT_NAME}        SHARED)

# add source files to each target
TARGET_SOURCES (${CMAKE_PROJECT_NAME}-static PRIVATE ${CORE_SRC} ${OP_SRC})
TARGET_SOURCES (${CMAKE_PROJECT_NAME}        PRIVATE ${CORE_SRC} ${OP_SRC})

#set .a version
set_target_properties(${CMAKE_PROJECT_NAME}-static PROPERTIES VERSION 0.0.0 SOVERSION 0)

#set .so version
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES VERSION 0.0.0 SOVERSION 0)

set_target_properties(${CMAKE_PROJECT_NAME}-static PROPERTIES OUTPUT_NAME "eutopia")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME "eutopia")

# # install tengine libraries
INSTALL (TARGETS ${CMAKE_PROJECT_NAME}        LIBRARY DESTINATION lib)
INSTALL (TARGETS ${CMAKE_PROJECT_NAME}-static ARCHIVE DESTINATION lib)

# install header files
INSTALL (FILES ${OP_HEAD} ${CORE_HEAD} DESTINATION include/eutopia/)

add_subdirectory(tools)
