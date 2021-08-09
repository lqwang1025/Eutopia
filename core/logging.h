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
#include <memory>

namespace eutopia {
namespace core {

using log_stream_t = std::unique_ptr<std::ostream>;

enum LogLevel {
    kEmerg = 0,
    kAlert,
    kCrit,
    kError,
    kWarn,
    kNotice,
    kInfo,
    kDebug
};

struct LogOption {
    std::string prefix;
    int max_line_size;
    int rate_limit; /* KB per second */
    bool log_level;
    bool log_date;
};

struct Logger {
    virtual bool set_log_level(LogLevel level) = 0;
    virtual LogLevel get_log_level(void) = 0;

    /*option part */
    virtual bool set_log_option(const LogOption& opt) = 0;

    virtual LogOption get_log_option(void) = 0;

    virtual void set_log_output_func(void (*func)(const char*)) = 0;

    /*Log part */
    virtual log_stream_t Log(LogLevel) = 0;

    virtual ~Logger(){};

    static Logger* get_logger(); /* get the global logger */
    static void set_logger(Logger* log); /* set the global logger */
    static const char* log_level_str(LogLevel level);
};

/* the wrapper for other to use log utilities */

#define SET_LOG_OUTPUT(func) Logger::get_logger()->set_log_output_func(func)

#define DO_LOG(level) (*Logger::get_logger()->Log(level))
#define GET_LOG_OPTION() Logger::get_logger()->get_log_option()
#define SET_LOG_OPTION(opt) Logger::get_logger()->set_log_option(opt)
#define GET_LOG_LEVEL() Logger::get_logger()->get_log_level()
#define SET_LOG_LEVEL(l) Logger::get_logger()->set_log_level(l)

#define LOG_DEBUG() DO_LOG(kDebug)
#define LOG_INFO() DO_LOG(kInfo)
#define LOG_WARN() DO_LOG(kWarn)
#define LOG_ERROR() DO_LOG(kError)
#define LOG_ALERT() DO_LOG(kAlert)
#define LOG_FATAL() DO_LOG(kCrit)

#define XLOG_DEBUG() LOG_DEBUG() << __FILE__ << ":" << __LINE__ << " "
#define XLOG_INFO() LOG_INFO() << __FILE__ << ":" << __LINE__ << " "
#define XLOG_WARN() LOG_WARN() << __FILE__ << ":" << __LINE__ << " "
#define XLOG_ERROR() LOG_ERROR() << __FILE__ << ":" << __LINE__ << " "
#define XLOG_ALERT() LOG_ALERT() << __FILE__ << ":" << __LINE__ << " "
#define XLOG_FATAL() LOG_FATAL() << __FILE__ << ":" << __LINE__ << " "



} // namespace core
} // namespace eutopia

#endif /* __LOGGING_H__ */

