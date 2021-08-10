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
 * File       : logging.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-08:16:15:34
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __LOGGING_H__
#define __LOGGING_H__

#include <iostream>
#include <string>
#include <assert.h>

namespace eutopia {
namespace core {

#define RESET   "\033[0m\n"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#define EU_LOG std::cout<<BOLDGREEN
#define EU_ERROR std::cout<<BOLDRED
#define EU_WARN std::cout<<BOLDYELLOW
#define EU_ENDL RESET

#ifdef DEBUG
#define EU_ASSERT(x)                                            \
    do {                                                        \
        if (!(x)) {                                             \
            EU_ERROR<<"Assert failed: "<< #x<<__FILE<<EU_ENDL;  \
            assert(0);                                          \
        }                                                       \
    } while (false)
#else
#define EU_ASSERT(x)
#endif

#define CHECK(success, log)                                             \
    if(!(success)){                                                     \
        EU_ERROR<<"Check failed: "<< #success <<"==>"<< #log <<EU_ENDL;  \
        exit(0);                                                        \
    }

} // namespace core
} // namespace eutopia

#endif /* __LOGGING_H__ */

