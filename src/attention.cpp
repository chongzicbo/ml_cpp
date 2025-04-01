#include "attention.h"
#include <iostream>
#include <cmath>
#include <algorithm>

ScaledDotProductAttention::ScaledDotProductAttention(int d_model, int d_k, int d_v, int h) : d_model(d_model), d_k(d_k), d_v(d_v), h(h), rng(std::random_device{}())
{
    // Initialize weight matrices
    W_q.resize(h);
    W_k.resize(h);
    W_v.resize(h);

    for (int i = 0; i < h; i++)
    {
        W_q[i].resize(d_model);
        W_k[i].resize(d_model);
        W_v[i].resize(d_model);
        for (int j = 0; j < d_model; j++)
        {
            W_q[i][j] = vector<float>(d_k);
            W_k[i][j] = vector<float>(d_k);
            W_v[i][j] = vector<float>(d_v);
            std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
            for (int k = 0; k < d_k; k++)
            {
                W_q[i][j][k] = dist(rng);
                W_k[i][j][k] = dist(rng);
            }
            for (int k = 0; k < d_v; k++)
            {
                W_v[i][j][k] = dist(rng);
            }
        }
    }
    // Initialize output weight matrix
    W_o = initializeWeights(h * d_v, d_model);
}

vector<vector<float>> ScaledDotProductAttention::initializeWeights(int rows, int cols)
{
    vector<vector<float>> weights(rows, vector<float>(cols));
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            weights[i][j] = dist(rng);
        }
    }
    return weights;
}

vector<vector<float>> ScaledDotProductAttention::matmul(const vector<vector<float>> &A, const vector<vector<float>> &B)
{
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty() || A[0].size() != B.size())
    {
        std::cerr << "Error: Invalid matrix dimensions\n";
        return {};
    }
    int m = A.size();                                   // 行
    int n = A[0].size();                                // 列
    int p = B[0].size();                                // 列
    vector<vector<float>> C(m, vector<float>(p, 0.0f)); // 创建一个m行p列的二维数组，并初始化为0
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

vector<vector<float>> ScaledDotProductAttention::transpose(const vector<vector<float>> &matrix)
{
    if (matrix.empty() || matrix[0].empty())
    {
        throw std::runtime_error("Error: cannot transpose empty matrix\n");
    }
    int m = matrix.size();
    int n = matrix[0].size();
    vector<vector<float>> A_T(n, vector<float>(m, 0.0f));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A_T[j][i] = matrix[i][j];
        }
    }
    return A_T;
}

vector<float> ScaledDotProductAttention::softmax(const vector<float> &x)
{
    vector<float> result(x.size());
    float max = *max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (int i = 0; i < x.size(); i++)
    {
        result[i] = exp(x[i] - max);
        sum += result[i];
    }
    for (int i = 0; i < x.size(); i++)
    {
        result[i] /= sum;
    }
    return result;
}
vector<vector<float>> ScaledDotProductAttention::softmaxRows(const vector<vector<float>> &A)
{
    if (A.empty() || A[0].empty())
    {
        throw std::runtime_error("Error: cannot softmax empty matrix\n");
    }
    vector<vector<float>> result(A.size(), vector<float>(A[0].size()));
    for (int i = 0; i < A.size(); i++)
    {
        if (A[i].size() != A[0].size())
            throw std::runtime_error("Error: matrix is not square");
        result[i] = softmax(A[i]);
    }
    return result;
}
vector<vector<float>> ScaledDotProductAttention::concatenateHorizontal(const vector<vector<vector<float>>> &matrices)
{
    if (matrices.empty())
    {
        throw std::runtime_error("Error: cannot concatenate empty list of matrices");
    }
    int m = matrices[0].size();
    int total_cols = 0;
    for (const auto &matrix : matrices)
    {
        total_cols += matrix[0].size();
    }
    vector<vector<float>> result(m, vector<float>(total_cols));
    int col_offset = 0;
    for (const auto &matrix : matrices)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < matrix[0].size(); j++)
            {
                result[i][col_offset + j] = matrix[i][j];
            }
        }
        col_offset += matrix[0].size();
    }
    return result;
}
std::vector<std::vector<float>> ScaledDotProductAttention::forward(
    const std::vector<std::vector<float>> &X)
{

    // 获取输入序列的长度
    int seq_len = X.size();

    // Store attention outputs from each head
    std::vector<std::vector<std::vector<float>>> head_outputs(h);

    // Process each attention head
    for (int i = 0; i < h; i++)
    {
        // Project input to query, key, and value
        std::vector<std::vector<float>> Q(seq_len, std::vector<float>(d_k, 0.0f));
        std::vector<std::vector<float>> K(seq_len, std::vector<float>(d_k, 0.0f));
        std::vector<std::vector<float>> V(seq_len, std::vector<float>(d_v, 0.0f));

        // Perform matrix multiplication with 3D weight matrices
        for (int j = 0; j < seq_len; j++)
        {
            for (int k = 0; k < d_k; k++)
            {
                for (int m = 0; m < d_model; m++) //  遍历模型维度d_model，计算Q和K矩阵的元素
                {
                    Q[j][k] += X[j][m] * W_q[i][m][k]; //  计算Q矩阵的第j行第k列的元素
                    K[j][k] += X[j][m] * W_k[i][m][k]; //  计算K矩阵的第j行第k列的元素
                }
            }
            for (int k = 0; k < d_v; k++) //  遍历输出维度d_v，计算V矩阵的元素
            {
                for (int m = 0; m < d_model; m++) //  遍历模型维度d_model
                {
                    V[j][k] += X[j][m] * W_v[i][m][k]; //  计算V矩阵的第j行第k列的元素
                }
            }
        }

        // Calculate attention scores: Q * K^T / sqrt(d_k)
        std::vector<std::vector<float>> K_T = transpose(K);      // [d_k][seq_len] //  将矩阵K进行转置操作，转置后的矩阵维度为[d_k][seq_len]
        std::vector<std::vector<float>> scores = matmul(Q, K_T); // [seq_len][seq_len] //  计算矩阵Q和转置后的矩阵K_T的乘积，结果矩阵维度为[seq_len][seq_len]

        // Scale scores
        float scale = std::sqrt(d_k);
        for (int j = 0; j < seq_len; j++) //  遍历序列长度，对scores矩阵中的每个元素进行缩放
        {
            for (int k = 0; k < seq_len; k++)
            {
                scores[j][k] /= scale; //  将scores矩阵中的每个元素除以scale值
            }
        }

        // Apply softmax to get attention weights
        std::vector<std::vector<float>> attention_weights = softmaxRows(scores); // [seq_len][seq_len] //  使用softmax函数对分数矩阵的每一行进行归一化，得到注意力权重矩阵         注意力权重矩阵的维度为[seq_len][seq_len]，其中seq_len是序列长度

        // Apply attention weights to values
        std::vector<std::vector<float>> head_output = matmul(attention_weights, V); // [seq_len][d_v] //  计算注意力机制的输出，通过将注意力权重与V矩阵相乘         head_output的维度为[seq_len][d_v]，其中seq_len是序列长度，d_v是每个头的向量维度
        head_outputs[i] = head_output;
    }

    // Concatenate outputs from all heads
    std::vector<std::vector<float>> concatenated = concatenateHorizontal(head_outputs); // [seq_len][h*d_v] //  将head_outputs按水平方向拼接成一个二维向量，其中每个子向量表示一个时间步的输出     结果的维度为[seq_len][h*d_v]，其中seq_len是序列长度，h是头的数量，d_v是每个头的向量维度

    // Project back to original dimension
    std::vector<std::vector<float>> output = matmul(concatenated, W_o); // [seq_len][d_model] //  定义一个二维向量output，用于存储矩阵乘法的结果     matmul函数执行矩阵乘法，concatenated和W_o是输入矩阵     结果的维度为[seq_len][d_model]，其中seq_len是序列长度，d_model是模型维度

    return output;
}

void ScaledDotProductAttention::printMatrix(
    const std::vector<std::vector<float>> &matrix,
    const std::string &name)
{

    std::cout << name << " (" << matrix.size() << "x"
              << (matrix.empty() ? 0 : matrix[0].size()) << "):\n";

    for (const auto &row : matrix)
    {
        for (float val : row)
        {
            std::cout << val << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}