#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "linear_elnet.hpp"

TEST(LinearElnet, LinearLassoOptim)
{
    Eigen::Matrix3d X;
    X << -0.48, -1.33, 1.22,
        -0.90, 0.26, 0,
        1.39, 1.07, -1.22;
    Eigen::Vector3d y;
    y << -1.12, 1.30, -0.18;
    const double lambda = 10;
    const double n_vars = 3;
    const double n_obs = 3;
    Eigen::Vector3d init_beta;
    init_beta << 0, 0, 0;
    Eigen::Vector3d expected_beta;
    expected_beta << 0, 0, 0;
    Eigen::VectorXd out_beta = linear_lasso_optim(X, y, lambda, n_vars, n_obs, init_beta);
    EXPECT_EQ(out_beta, expected_beta);
}

TEST(LinearElnet, LassoRuns)
{
    Eigen::Matrix3d X;
    X << -0.48, -1.33, 1.22,
        -0.90, 0.26, 0,
        1.39, 1.07, -1.22;
    Eigen::Vector3d y;
    y << -1.12, 1.30, -0.18;
    Eigen::VectorXd lambdas(5);
    lambdas << 0, 0.5, 0.8, 0.9, 1;
    Eigen::MatrixXd out_beta_mat = linear_elnet_coefs(X, y, lambdas, 1);
}

TEST(LinearElnet, RidgeRuns)
{
    Eigen::Matrix3d X;
    X << -0.48, -1.33, 1.22,
        -0.90, 0.26, 0,
        1.39, 1.07, -1.22;
    Eigen::Vector3d y;
    y << -1.12, 1.30, -0.18;
    Eigen::VectorXd lambdas(6);
    lambdas << 0, 0.5, 0.8, 0.9, 1, 100;
    Eigen::MatrixXd out_beta_mat = linear_elnet_coefs(X, y, lambdas, 0);
}

TEST(LinearElnet, ElasticNetRuns)
{
    Eigen::Matrix3d X;
    X << -0.48, -1.33, 1.22,
        -0.90, 0.26, 0,
        1.39, 1.07, -1.22;
    Eigen::Vector3d y;
    y << -1.12, 1.30, -0.18;
    Eigen::VectorXd lambdas(5);
    lambdas << 0, 0.5, 0.8, 0.9, 1;
    Eigen::MatrixXd out_beta_mat = linear_elnet_coefs(X, y, lambdas, 0.77);
}
