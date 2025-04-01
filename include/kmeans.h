#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "point.h"

class KMeans
{
public:
    // 构造函数，初始化k值和最大迭代次数
    KMeans(int k, int max_iterations = 100);
    // 使用数据集进行k-means聚类
    void fit(const std::vector<Point> &data);
    // 获取每个数据点所属的簇标签
    const std::vector<int> &get_labels() const { return labels_; }
    // 获取每个簇的中心点
    const std::vector<Point> &get_centers() const { return centers_; }

private:
    // 簇的数量
    int k_;
    // 最大迭代次数
    int max_iterations_;
    // 存储每个数据点所属的簇标签
    std::vector<int> labels_;
    // 存储每个簇的中心点
    std::vector<Point> centers_;

    // 计算两个点之间的欧氏距离
    double distance(const Point &a, const Point &b) const;
    // 初始化簇的中心点
    void initialize_centers(const std::vector<Point> &data);
    // 将数据点分配到最近的簇
    void assign_clusters(const std::vector<Point> &data);
    // 更新每个簇的中心点
    void update_centers(const std::vector<Point> &data);
    // 检查算法是否收敛
    bool has_converged(const std::vector<Point> &old_centers) const;
};

#endif // KMEANS_H