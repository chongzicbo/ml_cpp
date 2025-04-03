/*
Created by chengbo on 2025/4/2 17:01
email: chengbocbo@163.com
*/
#ifndef TRANSFORMER_RELU_HPP
#define TRANSFORMER_RELU_HPP

#include <array>

namespace transformer {
    template<typename T, int DIM>
    class Relu {
    public:
        static void forward(std::array<T, DIM> &input, std::array<T, DIM> &output) {
            for (int i = 0; i < DIM; ++i) {
                if (input[i] < 0) {
                    output[i] = 0;
                } else {
                    output[i] = input[i];
                }
            }
        }
    };

}
#endif