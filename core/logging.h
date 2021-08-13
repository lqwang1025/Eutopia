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
#include <cstdlib>

namespace eutopia {
namespace core {

#define RESET   "\033[0m\n"
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */

#define EU_LOG std::cout<<BOLDGREEN<<"Log: "
#define EU_ERROR std::cout<<BOLDRED<<"Error: "
#define EU_WARN std::cout<<BOLDYELLOW<<"Warning: "
#define EU_ENDL RESET

#ifdef DEBUGEUTOPIA
#define CHECK(success, log)                                             \
    if(!(success)){                                                     \
        EU_ERROR<<"Check failed: "<< #success <<" ==>["<<__FILE__<<": " \
                <<__LINE__<<"]"<< #log <<EU_ENDL;                       \
        abort();                                                        \
    }
#else
#define CHECK(success, log)                                             \
    if(!(success)){                                                     \
        EU_ERROR<<"Check failed: "<< #success <<"==>"<< #log <<EU_ENDL; \
        abort();                                                        \
    }
#endif


} // namespace core
} // namespace eutopia

#endif /* __LOGGING_H__ */

