#include "kmeans.h"
#include "gradient_descent.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include "attention.h"
#include "self_attention.hpp"

std::vector<Point> load_data(const std::string &filename) {
    std::vector<Point> data;
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'\n";
        return data;
    }
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        Point point;
        ss >> point.x >> point.y;
        data.push_back(point);
    }

    return data;
}

void test_kmeans() { // 加载数据
    std::vector<Point> data = load_data("/data/bocheng/dev/mylearn/cplus/ml_cpp/data/points.txt");

    // 设置 K 值
    int k = 3;

    // 创建 KMeans 对象并训练
    KMeans kmeans(k);
    kmeans.fit(data);

    // 输出结果
    const auto &labels = kmeans.get_labels();
    const auto &centers = kmeans.get_centers();

    std::cout << "Cluster Centers:\n";
    for (const auto &center: centers) {
        std::cout << "(" << center.x << ", " << center.y << ")\n";
    }

    std::cout << "\nData Points and Labels:\n";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << "(" << data[i].x << ", " << data[i].y << ") -> Cluster "
                  << labels[i] << "\n";
    }
}

void test_sgd() {
    std::vector<Point> data = load_data("/data/bocheng/dev/mylearn/cplus/ml_cpp/data/points.txt");
    if (data.empty()) {
        return;
    }
    GradientDescent sgd(0.01, 100);
    std::cout << "start sgd" << std::endl;
    sgd.batch_gradient_descent(data);
    std::cout << "Slope: " << sgd.get_slope() << std::endl;
    std::cout << "Intercept: " << sgd.get_intercept() << std::endl;
}

void test_autodiff();

void test_attention() {
    int d_model = 4; // Input dimension
    int d_k = 4;     // Key dimension
    int d_v = 4;     // Value dimension
    int h = 2;       // Number of attention heads
    int seq_len = 3; // Sequence length

    // Create input sequence
    std::vector<std::vector<float>> X = {
            {1.0f, 0.0f, 1.0f, 0.0f}, // Token 1 embedding
            {0.0f, 1.0f, 0.0f, 1.0f}, // Token 2 embedding
            {1.0f, 1.0f, 0.0f, 0.0f}  // Token 3 embedding
    };

    std::cout << "Testing Scaled Dot-Product Attention:\n";
    // Create scaled dot-product attention module
    ScaledDotProductAttention attention(d_model, d_k, d_v, h);

    // Print input
    ScaledDotProductAttention::printMatrix(X, "Input");

    // Apply scaled dot-product attention
    std::vector<std::vector<float>> output1 = attention.forward(X);

    // Print output
    ScaledDotProductAttention::printMatrix(output1, "Scaled Dot-Product Attention Output");
}


void test_self_attention() {
    try {
        // 测试数据准备
        int batch_size = 2;
        int d_model = 8;
        int d_k = 4;
        int d_v = 4;
        int h = 2;

        // 创建输入矩阵 (batch_size x d_model)
        std::vector<std::vector<float>> input(batch_size, std::vector<float>(d_model));
        input[0] = {1.0f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        input[1] = {0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f};

        // 创建SelfAttention实例
        SelfAttention sa(d_model, d_k, d_v, h);

        // 打印权重矩阵
        std::cout << "Initialized weights:" << std::endl;
        SelfAttention::printMatrix(sa.get_W_q()[0], "W_q head 0");
        SelfAttention::printMatrix(sa.get_W_k()[0], "W_k head 0");
        SelfAttention::printMatrix(sa.get_W_v()[0], "W_v head 0");
        SelfAttention::printMatrix(sa.get_W_o(), "W_o");

        // 前向传播
        auto output = sa.forward(input);

        // 打印输出
        std::cout << "\nOutput:" << std::endl;
        SelfAttention::printMatrix(output, "Output");
    }
    catch (const std::exception &e) {
        std::cerr << "Error in test_self_attention: " << e.what() << std::endl;
    }
}

int main() {
    // test_kmeans();
//    test_attention();
    test_self_attention();
    return 0;

}