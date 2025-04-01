#ifndef ATTENTION_H
#define ATTENTION_H

#include <vector>
#include <random>
using std::vector;
class ScaledDotProductAttention
{

private:
    int d_model;
    int d_k;
    int d_v;
    int h;

    vector<vector<vector<float>>> W_q; // （h,d_model,d_k）
    vector<vector<vector<float>>> W_k; // （h,d_model,d_k）
    vector<vector<vector<float>>> W_v; // （h,d_model,d_v）
    vector<vector<float>> W_o;         // (h*d_v,d_model)

    std::mt19937 rng;

    vector<vector<float>> initializeWeights(int rows, int cols);
    vector<vector<float>> matmul(const vector<vector<float>> &A, const vector<vector<float>> &B);
    vector<vector<float>> transpose(const vector<vector<float>> &A);
    vector<float> softmax(const vector<float> &A);
    vector<vector<float>> softmaxRows(const vector<vector<float>> &A);
    vector<vector<float>> concatenateHorizontal(const vector<vector<vector<float>>> &matrices);

public:
    ScaledDotProductAttention(int d_model, int d_k, int d_v, int h);
    vector<vector<float>> forward(const vector<vector<float>> &X);
    static void printMatrix(const vector<vector<float>> &matrix, const std::string &name);
};

#endif