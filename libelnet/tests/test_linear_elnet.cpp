#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "linear_elnet.hpp"

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

TEST(LinearElnet, EarlyStopping)
{
    Eigen::Matrix3d X;
    X << -0.48, -1.33, 1.22,
        -0.90, 0.26, 0,
        1.39, 1.07, -1.22;
    Eigen::Vector3d y;
    y << -1.12, 1.30, -0.18;
    Eigen::VectorXd lambdas(5);
    lambdas << 0, 0.5, 0.8, 0.9, 1;
    int expected_rows = 3;
    int expected_cols = 5;
    Eigen::MatrixXd out_beta_mat = linear_elnet_coefs(X, y, lambdas, 0.77, false);
    int beta_rows = out_beta_mat.rows();
    int beta_cols = out_beta_mat.cols();
    EXPECT_EQ(beta_rows, expected_rows);
    EXPECT_EQ(beta_cols, expected_cols);
}
