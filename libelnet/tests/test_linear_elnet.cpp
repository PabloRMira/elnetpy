#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "linear_elnet.hpp"

TEST(LinearElnet, LinearElnetOptim)
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

TEST(LinearElnet, LassoCoefsNumber)
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
    int expected_nrow = 3;
    int expected_ncol = 5;
    int out_nrow = out_beta_mat.rows();
    int out_ncol = out_beta_mat.cols();
    EXPECT_EQ(out_nrow, expected_nrow);
    EXPECT_EQ(out_ncol, expected_ncol);
}


TEST(LinearElnet, RidgeCoefsNumber)
{
    Eigen::Matrix3d X;
    X << -0.48, -1.33, 1.22,
        -0.90, 0.26, 0,
        1.39, 1.07, -1.22;
    Eigen::Vector3d y;
    y << -1.12, 1.30, -0.18;
    Eigen::VectorXd lambdas(5);
    lambdas << 0, 0.5, 0.8, 0.9, 1;
    Eigen::MatrixXd out_beta_mat = linear_elnet_coefs(X, y, lambdas, 0);
    int expected_nrow = 3;
    int expected_ncol = 5;
    int out_nrow = out_beta_mat.rows();
    int out_ncol = out_beta_mat.cols();
    EXPECT_EQ(out_nrow, expected_nrow);
    EXPECT_EQ(out_ncol, expected_ncol);
}


TEST(LinearElnet, ElasticNetCoefsNumber)
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
    int expected_nrow = 3;
    int expected_ncol = 5;
    int out_nrow = out_beta_mat.rows();
    int out_ncol = out_beta_mat.cols();
    EXPECT_EQ(out_nrow, expected_nrow);
    EXPECT_EQ(out_ncol, expected_ncol);
}
