#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    // 构造函数需要，cost function（约束），loss function：残差的计算方式，相关联的参数块，待边缘化的参数块的索引
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    // 优化变量数据
    std::vector<double *> parameter_blocks;
    // 待marg的优化变量id
    std::vector<int> drop_set;

    //Jacobian
    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;  // Eigen的动态矩阵
    // 残差，IMU：15×1,视觉：2×1
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    // 加残差块相关信息(优化变量、待marg的变量)
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    // 计算每个残差对应的Jacobian，并更新parameter_block_data
    void preMarginalize();
    // pos为所有变量维度，m为需要marg掉的变量，n为需要保留的变量
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    // 所有观测项
    std::vector<ResidualBlockInfo *> factors;
    // m为要marg掉的变量个数，也就是parameter_block_idx的总localSize，以double为单位，VBias为9，PQ为6，
    // n为要保留下的优化变量的变量个数，n=localSize(parameter_block_size) – m
    int m, n;
    // <优化变量内存地址，localSize>
    std::unordered_map<long, int> parameter_block_size; //global size   // 地址->global size
    int sum_block_size;
    // <待marg的优化变量内存地址，在parameter_block_size中的id,以double为单位>
    std::unordered_map<long, int> parameter_block_idx; //local size // 地址->参数排列的顺序idx
    // <优化变量内存地址，数据>
    std::unordered_map<long, double *> parameter_block_data;    // 地址->参数块实际内容的地址

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

// 由于边缘化的costfuntion不是固定大小的，因此只能继承最基本的类
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
