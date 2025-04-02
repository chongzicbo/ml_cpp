/*
Created by chengbo on 2025/4/2 9:41
email: chengbocbo@163.com
*/
#include "eigen_self_attention.hpp"
#include <stdexcept>
namespace EigenSelfAttention{
    SelfAttention::SelfAttention(int embed_dim, int num_heads):embed_dim_(embed_dim),num_heads_(num_heads) {
        if(embed_dim % num_heads != 0)
        {
            throw std::invalid_argument("embed_dim must be divisible by num_heads");
        }
        head_dim_ = embed_dim / num_heads;
        W_q_= Eigen::MatrixXd::Random(embed_dim, embed_dim);
        W_k_= Eigen::MatrixXd::Random(embed_dim, embed_dim);
        W_v_= Eigen::MatrixXd::Random(embed_dim, embed_dim);
        W_o_= Eigen::MatrixXd::Random(embed_dim, embed_dim);

    }
    Eigen::MatrixXd  SelfAttention::softmax(const Eigen::MatrixXd &input) const {
        Eigen::MatrixXd exp_input=input.array().exp();
        Eigen::VectorXd row_sums=exp_input.rowwise().sum();
        return exp_input.array().colwise()/row_sums.array();
    }
    Eigen::MatrixXd  SelfAttention::splitHeads(const Eigen::MatrixXd &input) const {
        int batch_size=input.rows();
        Eigen::MatrixXd reshaped(batch_size*num_heads_,head_dim_);
        for(int i=0;i<batch_size;i++)
        {
            for(int j=0;j<num_heads_;j++)
            {
                reshaped.block(i*num_heads_+j,0,1,head_dim_)=input.block(i,j*head_dim_,1,head_dim_);
            }
        }
        return reshaped;
    }

    Eigen::MatrixXd SelfAttention::combineHeads(const Eigen::MatrixXd &input) const
    {
        int batch_size = input.rows() / num_heads_;
        Eigen::MatrixXd combined(batch_size, embed_dim_);
        for (int i = 0; i < batch_size; ++i)
        {
            for (int j = 0; j < num_heads_; ++j)
            {
                combined.block(i, j * head_dim_, 1, head_dim_) =
                        input.block(i * num_heads_ + j, 0, 1, head_dim_);
            }
        }
        return combined;
    }

    // 前向传播
    Eigen::MatrixXd SelfAttention::forward(const Eigen::MatrixXd &input)
    {
        int batch_size = input.rows();

        // 计算 Q, K, V
        Eigen::MatrixXd Q = input * W_q_;
        Eigen::MatrixXd K = input * W_k_;
        Eigen::MatrixXd V = input * W_v_;

        // 分割头
        Eigen::MatrixXd Q_split = splitHeads(Q);
        Eigen::MatrixXd K_split = splitHeads(K);
        Eigen::MatrixXd V_split = splitHeads(V);

        // Scaled Dot-Product Attention
        Eigen::MatrixXd scores = (Q_split * K_split.transpose()) / std::sqrt(static_cast<double>(head_dim_));
        Eigen::MatrixXd attention_weights = softmax(scores);
        Eigen::MatrixXd output = attention_weights * V_split;

        // 合并头
        Eigen::MatrixXd output_combined = combineHeads(output);

        // 输出线性变换
        return output_combined * W_o_;
    }
}