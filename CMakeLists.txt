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

# Enable the languages which in use
ENABLE_LANGUAGE (C CXX)

IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
ENDIF()

# Project name
PROJECT (eutopia)

# Enable cmake ctest module
ENABLE_TESTING ()

# clear all var
UNSET (EUTOPIA_COMMON_OPTIONS)
UNSET (EUTOPIA_COMMON_LINKER_OPTIONS)

IF (${CMAKE_BUILD_TYPE} MATCHES "Check" OR ${CMAKE_BUILD_TYPE} MATCHES "CHECK")
  # warning local redefined same name variable
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-Wshadow")
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-Wunreachable-code")
  # LIST (APPEND EUTOPIA_COMMON_OPTIONS "-Werror")

  # show all warnings
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-Wall")

  # more warning details
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-Wextra")

  # warning all unused part
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-Wunknown-pragmas")
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-Wunused-label")
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-Wunused-value")
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-Wunused-function")
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-Wunused-variable")

  # check memory issues
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-fsanitize=address")
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-fno-omit-frame-pointer")

  LIST (APPEND EUTOPIA_COMMON_LINKER_OPTIONS "-fsanitize=address")
ELSEIF(${CMAKE_BUILD_TYPE} MATCHES "Debug" OR ${CMAKE_BUILD_TYPE} MATCHES "DEBUG")
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-g")
  LIST (APPEND EUTOPIA_COMMON_OPTIONS "-O0")
ENDIF()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Main source files
ADD_SUBDIRECTORY (core)
ADD_SUBDIRECTORY (test)

# Report summary
INCLUDE ("${CMAKE_CURRENT_SOURCE_DIR}/cmake/summary.cmake")

# # install tengine libraries
# INSTALL (TARGETS ${CMAKE_PROJECT_NAME}        LIBRARY DESTINATION lib)
# INSTALL (TARGETS ${CMAKE_PROJECT_NAME}-static ARCHIVE DESTINATION lib)

# # install header files
# INSTALL (FILES ${EUTOPIA_H_FILE} DESTINATION include/eutopia/core)
