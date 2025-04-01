#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include <vector>
#include <cmath>
#include <random>
#include "point.h"

class GradientDescent
{
public:
    GradientDescent(double learning_rate, int max_iterations, double tolerance = 1e-6);
    void batch_gradient_descent(const std::vector<Point> &data);
    void stochastic_gradient_descent( std::vector<Point> &data);
    void mini_batch_gradient_descent( std::vector<Point> &data, int batch_size);

    // 获取训练结果
    double get_slope() const { return slope_; }
    double get_intercept() const { return intercept_; }

private:
    double learning_rate_;
    int max_iterations_;
    double tolerance_;
    double slope_;     // 斜率
    double intercept_; // 截距

    double compute_loss(const std::vector<Point> &data) const;
    void update_parameters(const std::vector<Point> &data, double &gradient_slope, double &gradient_intercept);
};
#endif // GRADIENT_DESCENT_H