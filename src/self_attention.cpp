/*
Created by chengbo on 2025/4/1 16:41
email: chengbocbo@163.com
*/
#include "self_attention.hpp"
#include <iostream>
#include <algorithm>

SelfAttention::SelfAttention(int d_model, int d_k, int d_v, int h, int max_seq_length):d_model(d_model), d_k(d_k), d_v(d_v), h(h), max_seq_length(max_seq_length),rng(std::random_device{}()) {
    std::cout << "SelfAttention init" << std::endl;
    W_q.resize(h);
    W_k.resize(h);
    W_v.resize(h);
    for(int i=0;i<h;i++)
    {
        W_q[i]= initialize_weights(d_model,d_k);
        W_k[i]=initialize_weights(d_model,d_k);
        W_v[i]=initialize_weights(d_model,d_v);
    }
    W_o = initialize_weights(d_v * h, d_model);
    initializePositionalEncoding();
}

vector<vector<float>> SelfAttention::initialize_weights(int rows, int cols) {
    vector<vector<float>> weights(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            weights[i][j] = std::normal_distribution<float>(-0.1f, 0.1f)(rng);
        }
    }
    return weights;
}

void SelfAttention::initializePositionalEncoding() {
    pos_embedding.resize(max_seq_length,vector<float>(d_model,0.0f));
    for (int i = 0; i < max_seq_length; ++i) {
        for (int j = 0; j < d_model; j+=2) {
            pos_embedding[i][j] = std::sin(i / std::pow(10000, (2.0*j) / d_model));
            if(j+1<d_model)
            {
                pos_embedding[i][j+1]= std::cos(i / std::pow(10000, (2.0*j) / d_model));
            }
        }
    }
}

vector<vector<float>> SelfAttention::addPositionalEncoding(const std::vector<std::vector<float>> &X) {
    int seq_len=X.size();
    vector<vector<float>> encoded=X;
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoded[i][j]+=pos_embedding[i][j];
        }
    }
    return encoded;
}

vector<vector<float>> SelfAttention::matmul(const std::vector<std::vector<float>> &A,
                                            const std::vector<std::vector<float>> &B) {

    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();

    std::vector<std::vector<float>> C(m, std::vector<float>(p, 0.0f));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

std::vector<std::vector<float>> SelfAttention::transpose(
        const std::vector<std::vector<float>>& A) {

    int m = A.size();
    int n = A[0].size();

    std::vector<std::vector<float>> A_T(n, std::vector<float>(m));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A_T[j][i] = A[i][j];
        }
    }

    return A_T;
}

std::vector<float> SelfAttention::softmax(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;

    for (size_t i = 0; i < x.size(); i++) {
        result[i] = std::exp(x[i] - max_val); // Subtract max for numerical stability
        sum += result[i];
    }

    for (size_t i = 0; i < x.size(); i++) {
        result[i] /= sum;
    }

    return result;
}

std::vector<std::vector<float>> SelfAttention::softmaxRows(
        const std::vector<std::vector<float>>& A) {

    std::vector<std::vector<float>> result(A.size(), std::vector<float>(A[0].size()));

    for (size_t i = 0; i < A.size(); i++) {
        std::vector<float> row_softmax = softmax(A[i]);
        result[i] = row_softmax;
    }

    return result;
}

std::vector<std::vector<float>> SelfAttention::concatenateHorizontal(
        const std::vector<std::vector<std::vector<float>>>& matrices) {

    if (matrices.empty()) return {};

    int rows = matrices[0].size();
    int total_cols = 0;

    for (const auto& matrix : matrices) {
        total_cols += matrix[0].size();
    }

    std::vector<std::vector<float>> result(rows, std::vector<float>(total_cols));

    int col_offset = 0;
    for (const auto& matrix : matrices) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < matrix[0].size(); j++) {
                result[i][col_offset + j] = matrix[i][j];
            }
        }
        col_offset += matrix[0].size();
    }

    return result;
}

std::vector<std::vector<float>> SelfAttention::forward(
        const std::vector<std::vector<float>>& X) {

    // Add positional encoding to input
    std::vector<std::vector<float>> encoded_X = addPositionalEncoding(X);
    int seq_len = encoded_X.size();

    // Store attention outputs from each head
    std::vector<std::vector<std::vector<float>>> head_outputs(h);

    // Process each attention head
    for (int i = 0; i < h; i++) {
        // Project input to query, key, and value
        std::vector<std::vector<float>> Q = matmul(encoded_X, W_q[i]); // [seq_len][d_k]
        std::vector<std::vector<float>> K = matmul(encoded_X, W_k[i]); // [seq_len][d_k]
        std::vector<std::vector<float>> V = matmul(encoded_X, W_v[i]); // [seq_len][d_v]

        // Calculate attention scores: Q * K^T / sqrt(d_k)
        std::vector<std::vector<float>> K_T = transpose(K); // [d_k][seq_len]
        std::vector<std::vector<float>> scores = matmul(Q, K_T); // [seq_len][seq_len]

        // Scale scores
        float scale = std::sqrt(d_k);
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < seq_len; k++) {
                scores[j][k] /= scale;
            }
        }

        // Apply softmax to get attention weights
        std::vector<std::vector<float>> attention_weights = softmaxRows(scores); // [seq_len][seq_len]

        // Apply attention weights to values
        std::vector<std::vector<float>> head_output = matmul(attention_weights, V); // [seq_len][d_v]
        head_outputs[i] = head_output;
    }

    // Concatenate outputs from all heads
    std::vector<std::vector<float>> concatenated = concatenateHorizontal(head_outputs); // [seq_len][h*d_v]

    // Project back to original dimension
    std::vector<std::vector<float>> output = matmul(concatenated, W_o); // [seq_len][d_model]

    return output;
}

void SelfAttention::printMatrix(
        const std::vector<std::vector<float>>& matrix,
        const std::string& name) {

    std::cout << name << " (" << matrix.size() << "x"
              << (matrix.empty() ? 0 : matrix[0].size()) << "):\n";

    for (const auto& row : matrix) {
        for (float val : row) {
            std::cout << val << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}