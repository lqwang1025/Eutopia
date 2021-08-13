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
 * File       : EutopiaTest.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-08:13:49:52
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __EUTOPIATEST_H__
#define __EUTOPIATEST_H__

#include <string>

namespace eutopia {
namespace test {

class EutopiaTestCase {
    friend class EutopiaTestSuite;
public:
    virtual ~EutopiaTestCase() = default;
    virtual bool run(int precision) = 0;
    
private:
    std::string name_;
};

class EutopiaTestSuite {
public:
    ~EutopiaTestSuite();
    static EutopiaTestSuite* get();
    void add(EutopiaTestCase* test, const char* name);
    static void run_all(int precision);
    static void run(const char* name, int precision);
    
private:
    static EutopiaTestSuite* g_instanse_;
    std::vector<EutopiaTestCase*> m_tests_;
};

template <class Case>
class EutopiaTestRegister {
public:
    EutopiaTestRegister(const char* name) {
        EutopiaTestSuite::get()->add(new Case, name);
    }
    
    ~EutopiaTestRegister() {}
};

#define EUTOPIA_TEST_REGISTER(Case, name) static EutopiaTestRegister<Case> __r_##Case(name)

} // namespace test
} // namespace eutopia

#endif /* __EUTOPIATEST_H__ */

