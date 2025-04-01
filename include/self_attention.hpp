/*
Created by chengbo on 2025/4/1 16:31
email: chengbocbo@163.com
*/
#ifndef SELF_ATTENTION_HPP
#define SELF_ATTENTION_HPP

#include <vector>
#include <random>
#include <cmath>

using std::vector;

class SelfAttention {
private:
    int d_model;
    int d_k;
    int d_v;
    int h;
    int max_seq_length;
    vector<vector<vector<float>>> W_q;
    vector<vector<vector<float>>> W_k;
    vector<vector<vector<float>>> W_v;
    vector<vector<float>> W_o;

    //position embedding
    vector<vector<float>> pos_embedding; //[max_seq_length, d_model]
    std::mt19937 rng;

    //Helper functions
    vector<vector<float>> initialize_weights(int rows, int cols);

    std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>> &A,
                                           const std::vector<std::vector<float>> &B);

    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>> &A);

    std::vector<float> softmax(const std::vector<float> &x);

    std::vector<std::vector<float>> softmaxRows(const std::vector<std::vector<float>> &A);

    std::vector<std::vector<float>> concatenateHorizontal(
            const std::vector<std::vector<std::vector<float>>> &matrices);

    void initializePositionalEncoding();

    std::vector<std::vector<float>> addPositionalEncoding(const std::vector<std::vector<float>> &X);

public:
    SelfAttention(int d_model, int d_k, int d_v, int h, int max_seq_length = 1000);

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>> &X);

    static void printMatrix(const std::vector<std::vector<float>> &matrix, const std::string &name);
    const std::vector<std::vector<std::vector<float>>>& get_W_q() const { return W_q; }
    const std::vector<std::vector<std::vector<float>>>& get_W_k() const { return W_k; }
    const std::vector<std::vector<std::vector<float>>>& get_W_v() const { return W_v; }
    const std::vector<std::vector<float>>& get_W_o() const { return W_o; }
};

#endif //SELF_ATTENTION_HPP