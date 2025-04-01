#include "kmeans.h"
#include <iostream>

// 定义KMeans类的构造函数
KMeans::KMeans(int k, int max_iterations)
    // 初始化列表，用于在构造函数中直接初始化成员变量
    : k_(k), max_iterations_(max_iterations) {}

// 定义KMeans类的distance方法，用于计算两个点之间的欧几里得距离
double KMeans::distance(const Point &a, const Point &b) const
{
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

// KMeans类的成员函数，用于初始化聚类中心
void KMeans::initialize_centers(const std::vector<Point> &data)
{
    // 创建一个随机设备，用于生成随机数种子
    std::random_device rd;
    // 使用随机设备生成一个梅森旋转引擎的随机数生成器
    std::mt19937 gen(rd());
    // 创建一个均匀整数分布，范围是从0到数据点数量的最后一个索引
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    // 清空当前的聚类中心集合
    centers_.clear();
    // 循环k次，生成k个随机索引，并将对应的点作为初始聚类中心
    for (int i = 0; i < k_; ++i)
    {
        // 生成一个随机索引
        // 将数据中对应索引的点添加到聚类中心集合中
        centers_.push_back(data[dis(gen)]);
    }
}

// KMeans类的成员函数，用于将数据点分配到最近的聚类中心
void KMeans::assign_clusters(const std::vector<Point> &data)
{
    // 调整labels_的大小以匹配数据点的数量
    labels_.resize(data.size());
    // 遍历每个数据点
    for (size_t i = 0; i < data.size(); ++i)
    {
        // 初始化最小距离为最大可能的double值
        double min_dist = std::numeric_limits<double>::max();
        // 初始化聚类标签为-1（表示未分配）
        int cluster = -1;

        // 遍历每个聚类中心
        for (int j = 0; j < k_; ++j)
        {
            // 计算当前数据点到当前聚类中心的距离
            double dist = distance(data[i], centers_[j]);
            // 如果当前距离小于已知的最小距离，则更新最小距离和聚类标签
            if (dist < min_dist)
            {
                min_dist = dist;
                cluster = j;
            }
        }
        // 将当前数据点的聚类标签设置为最近的聚类中心
        labels_[i] = cluster;
    }
}

// KMeans类的成员函数，用于更新聚类中心
void KMeans::update_centers(const std::vector<Point> &data)
{
    // 创建一个大小为k_的整数向量，用于记录每个聚类的点数，初始值为0
    std::vector<int> counts(k_, 0);
    // 创建一个大小为k_的点向量，用于存储新的聚类中心，初始值为(0, 0)
    std::vector<Point> new_centers(k_, {0, 0});

    // 遍历数据集中的每个点
    for (size_t i = 0; i < data.size(); ++i)
    {
        // 获取当前点所属的聚类标签
        int cluster = labels_[i];
        // 将当前点的坐标累加到对应聚类中心的新坐标中
        new_centers[cluster].x += data[i].x;
        new_centers[cluster].y += data[i].y;
        // 对应聚类的点数加1
        counts[cluster]++;
    }

    // 遍历每个聚类
    for (int i = 0; i < k_; ++i)
    {
        // 如果当前聚类中有点
        if (counts[i] > 0)
        {
            // 计算新的聚类中心坐标，即累加的坐标除以点数
            centers_[i].x = new_centers[i].x / counts[i];
            centers_[i].y = new_centers[i].y / counts[i];
        }
    }
}

// KMeans类的成员函数，用于判断聚类中心是否已经收敛
bool KMeans::has_converged(const std::vector<Point> &old_centers) const
{
    // 遍历所有的聚类中心
    for (int i = 0; i < k_; ++i)
    {
        // 计算旧聚类中心和新聚类中心之间的距离
        // 如果距离大于一个很小的阈值（1e-6），则认为聚类中心还未收敛
        if (distance(old_centers[i], centers_[i]) > 1e-6)
        {
            // 如果有任何一个聚类中心未收敛，则返回false
            return false;
        }
    }
    // 如果所有聚类中心都收敛，则返回true
    return true;
}

// KMeans类的fit函数，用于对数据进行K均值聚类
void KMeans::fit(const std::vector<Point> &data)
{
    // 初始化聚类中心
    initialize_centers(data);

    // 进行最大迭代次数的循环
    for (int iter = 0; iter < max_iterations_; ++iter)
    {
        // 将数据点分配到最近的聚类中心
        assign_clusters(data);

        // 保存当前的聚类中心
        std::vector<Point> old_centers = centers_;
        // 更新聚类中心
        update_centers(data);

        // 检查是否收敛，如果收敛则输出信息并退出循环
        if (has_converged(old_centers))
        {
            std::cout << "Converged at iteration " << iter + 1 << std::endl;
            break;
        }
    }
}