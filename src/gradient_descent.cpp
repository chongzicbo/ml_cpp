#include "gradient_descent.h"
#include <iostream>
#include <algorithm>
#include <random>

GradientDescent::GradientDescent(double learning_rate, int max_iteration, double tolerance)
    : learning_rate_(learning_rate), max_iterations_(max_iteration), tolerance_(tolerance), slope_(0.0), intercept_(0.0)
{
}

double GradientDescent::compute_loss(const std::vector<Point> &data) const
{
    double loss = 0.0;
    for (const auto &point : data)
    {
        double error = (slope_ * point.x + intercept_) - point.y;
        loss += error * error;
    }
    return loss / (2 * data.size());
}
void GradientDescent::update_parameters(const std::vector<Point> &data, double &grad_slope, double &grad_intercept)
{
    grad_slope = 0.0;
    grad_intercept = 0.0;
    for (const auto &point : data)
    {
        double error = (slope_ * point.x + intercept_) - point.y;
        grad_slope += error * point.x;
        grad_intercept += error;
    }
    grad_slope /= data.size();
    grad_intercept /= data.size();
    slope_ -= learning_rate_ * grad_slope;
    intercept_ -= learning_rate_ * grad_intercept;
}

void GradientDescent::batch_gradient_descent(const std::vector<Point> &data)
{
    for (int iter = 0; iter < max_iterations_; ++iter)
    {
        double grad_slope, grad_intercept;
        update_parameters(data, grad_slope, grad_intercept);
        double loss = compute_loss(data);
        if (std::abs(grad_slope) < tolerance_ && std::abs(grad_intercept) < tolerance_)
        {
            std::cout << "Batch GD converged at iteration  " << iter + 1 << std::endl;
            break;
        }
        if (iter % 10 == 0)
        {
            std::cout << "Iteration " << iter + 1 << " loss: " << loss << std::endl;
        }
    }
}

void GradientDescent::stochastic_gradient_descent( std::vector<Point> &data)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int iter = 0; iter < max_iterations_; ++iter)
    {
        std::shuffle(data.begin(), data.end(), gen);
        for (const auto &point : data)
        {
            double grad_slope = ((slope_ * point.x + intercept_) - point.y) * point.x;
            double grad_intercept = (slope_ * point.x + intercept_) - point.y;
            slope_ -= learning_rate_ * grad_slope;
            intercept_ -= learning_rate_ * grad_intercept;
        }
        double loss = compute_loss(data);
        if (iter % 10 == 0)
        {
            std::cout << "Iteration " << iter + 1 << " loss: " << loss << std::endl;
        }
    }
}

void GradientDescent::mini_batch_gradient_descent( std::vector<Point> &data, int batch_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int iter = 0; iter < max_iterations_; ++iter)
    {
        std::shuffle(data.begin(), data.end(), gen);

        for (size_t i = 0; i < data.size(); i += batch_size)
        {
            std::vector<Point> batch(data.begin() + i, data.begin() + std::min(i + batch_size, data.size()));
            double grad_slope, grad_intercept;
            update_parameters(batch, grad_slope, grad_intercept);
        }

        double cost = compute_loss(data);
        if (iter % 10 == 0)
        {
            std::cout << "Iteration " << iter + 1 << ": Cost = " << cost << "\n";
        }
    }
}