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
 * File       : data_type.cpp
 * File       : filler.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-25:14:57:09
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __FILLER_H__
#define __FILLER_H__

namespace eutopia {
    
namespace core {
namespace ir {
class Tensor;
}
}

namespace utils {

class Filler {
public:
    Filler();
    virtual ~Filler() = default;
    virtual void fill(core::ir::Tensor* tensor) = 0;
    void set_seed(int seed);
    void remove_seed();
    int get_seed() const;
protected:
    int seed_;
    bool have_seed_;
};

class ConstantFiller : public Filler {
public:
    ConstantFiller(float value) : constant_value_(value){}

    virtual void fill(core::ir::Tensor* tensor) override;

private:
    float constant_value_;
};

class UniformFiller : public Filler {
public:
    UniformFiller(float min = 0.0f, float max = 1.0f) : min_(min), max_(max) {}

    virtual void fill(core::ir::Tensor* tensor) override;
private:
    float min_;
    float max_;
};

class NormalFiller : public Filler {
public:
    NormalFiller(float mean = 0.0f, float std = 1.0f) : mean_(mean), std_(std){}
    virtual void fill(core::ir::Tensor* tensor) override;

private:
    float mean_;
    float std_;
};

class TruncatedNormalFiller : public Filler {
public:
    TruncatedNormalFiller(float mean = 0.0f, float std = 1.0f) : mean_(mean), std_(std) {}
    virtual void fill(core::ir::Tensor* tensor) override;

private:
    float mean_;
    float std_;
};

class XavierFiller : public Filler {
public:
    enum VarianceType {
        FANIN,
        FANOUT,
        AVERAGE
    };
    
    XavierFiller(VarianceType type = FANIN) : varianceType_(type) {}
    virtual void fill(core::ir::Tensor* tensor) override;
private:
    VarianceType varianceType_;
};

class MSRAFiller : public Filler {
public:
    enum VarianceType {
        FANIN,
        FANOUT,
        AVERAGE
    };

    MSRAFiller(VarianceType type = FANIN) : varianceType_(type) {}
    virtual void fill(core::ir::Tensor* tensor) override;
private:
    VarianceType varianceType_;
};

class PositiveUnitballFiller : public Filler {
public:
    PositiveUnitballFiller(){}
    virtual void fill(core::ir::Tensor* tensor) override;
};


} // namespace utils
} // namespace eutopia

#endif /* __FILLER_H__ */

