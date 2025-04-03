/*
Created by chengbo on 2025/4/2 17:04
email: chengbocbo@163.com
*/
#ifndef TRANSFORMER_SOFTMAX_HPP
#define TRANSFORMER_SOFTMAX_HPP

#include <array>

template<typename T, int DIM, int DEP>
class Softmax {
public:
    static void forward(std::array<std::array<T, DIM>, DEP> input, std::array<std::array<T, DIM>, DEP> &output) {
        for (int j = 0; j < DIM; ++j) {
            T tmp = 0;
            for (int i = 0; i < DEP; ++i) {
                tmp += input[i][j];
            }
            for (int i = 0; i < DEP; ++i) {
                output[i][j] = input[i][j] / tmp;
            }
        }
    }
};

#endif