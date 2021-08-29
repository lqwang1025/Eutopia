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
 * File       : avtivation.h
 * Authors    : Daniel Wang
 * Create Time: 2021-08-26:16:56:32
 * Email      : wangliquan21@qq.com
 * Description:
 */

#ifndef __AVTIVATION_H__
#define __AVTIVATION_H__
#include <cmath>
#include <string>

namespace eutopia {
namespace op {
namespace cpu {

typedef float(*ActivationFunc)(float);

ActivationFunc get_activation(const std::string& act_type);

static inline float stair_activate(float x) {
    int n = floor(x);
    if (n%2 == 0)
        return floor(x/2.);
    else
        return (x - n) + floor(x/2.);
}

static inline float hardtan_activate(float x) {
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}

static inline float linear_activate(float x) {
    return x;
}

static inline float logistic_activate(float x) {
    return 1./(1. + exp(-x));
}

static inline float loggy_activate(float x) {
    return 2./(1. + exp(-x)) - 1;
}

static inline float relu_activate(float x) {
    return x*(x>0);
}

static inline float elu_activate(float x) {
    return (x >= 0)*x + (x < 0)*(exp(x)-1);
}

static inline float selu_activate(float x) {
    return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);
}

static inline float relie_activate(float x) {
    return (x>0) ? x : .01*x;
}

static inline float ramp_activate(float x) {
    return x*(x>0)+.1*x;
}

static inline float leaky_activate(float x) {
    return (x>0) ? x : .1*x;
}

static inline float tanh_activate(float x) {
    return (exp(2*x)-1)/(exp(2*x)+1);
}

static inline float plse_activate(float x) {
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate(float x) {
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}


static inline float lhtan_gradient(float x) {
    if(x > 0 && x < 1) return 1;
    return .001;
}

static inline float hardtan_gradient(float x) {
    if (x > -1 && x < 1) return 1;
    return 0;
}

static inline float linear_gradient(float x) {
    return 1;
}

static inline float logistic_gradient(float x) {
    return (1-x)*x;
}

static inline float loggy_gradient(float x) {
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}

static inline float stair_gradient(float x) {
    if (floor(x) == x) return 0;
    return 1;
}

static inline float relu_gradient(float x) {
    return (x>0);
}

static inline float elu_gradient(float x) {
    return (x >= 0) + (x < 0)*(x + 1);
}

static inline float selu_gradient(float x) {
    return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);
}

static inline float relie_gradient(float x) {
    return (x>0) ? 1 : .01;
}

static inline float ramp_gradient(float x) {
    return (x>0)+.1;
}

static inline float leaky_gradient(float x) {
    return (x>0) ? 1 : .1;
}

static inline float tanh_gradient(float x) {
    return 1-x*x;
}

static inline float plse_gradient(float x) {
    return (x < 0 || x > 1) ? .01 : .125;
}


} // namespace cpu
} // namespace op
} // namespace eutopia

#endif /* __AVTIVATION_H__ */

