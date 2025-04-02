/*
Created by chengbo on 2025/4/2 9:36
email: chengbocbo@163.com
*/
#ifndef EIGEN_SELF_ATTENTION_HPP_
#define EIGEN_SELF_ATTENTION_HPP_

#include <Eigen//Dense>
#include <iostream>
namespace EigenSelfAttention {
    class SelfAttention{
        public:
        SelfAttention(int embed_dim,int num_heads);
        Eigen::MatrixXd forward(const Eigen::MatrixXd &input);
    private:
        int embed_dim_; //embedding 维度
        int num_heads_; //head number
        int head_dim_; //head embedding 维度
        Eigen::MatrixXd  W_q_;
        Eigen::MatrixXd  W_k_;
        Eigen::MatrixXd  W_v_;
        Eigen::MatrixXd  W_o_;//output weight
        Eigen::MatrixXd  softmax(const Eigen::MatrixXd &input) const;
        Eigen::MatrixXd  splitHeads(const Eigen::MatrixXd &input) const;
        Eigen::MatrixXd  combineHeads(const Eigen::MatrixXd &input) const;
    };
}

#endif