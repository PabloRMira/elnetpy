#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "logistic_elnet.hpp"
#include "utils.hpp"

TEST(LogisticElnet, GetWeightedInputs)
{
    Eigen::MatrixXd X(3, 3);
    X << 1, 5, 3,
        2, 3, 1,
        8, 5, 2;
    Eigen::VectorXd y(3);
    y << 1, 0, 1;
    Eigen::VectorXd beta(3);
    beta << -1, .5, 1;
    const double intercept = -.5;
    WeightedInput out = get_weighted_inputs(X, y, beta, intercept);
    Eigen::MatrixXd expected_Xt(3, 3);
    expected_Xt << 0.132901, 0.664506, 0.398703,
        1, 1.5, 0.5,
        1.06321, 0.664506, 0.265802;
    Eigen::VectorXd expected_yt(3);
    expected_yt << 0.66694, -1, 6.85745;
    EXPECT_TRUE(out.Xt.isApprox(expected_Xt, 1e-5));
    EXPECT_TRUE(out.yt.isApprox(expected_yt, 1e-5));
}

TEST(LogisticElnet, LassoRuns)
{
    Eigen::Matrix3d X;
    X << -0.48, -1.33, 1.22,
        -0.90, 0.26, 0,
        1.39, 1.07, -1.22;
    Eigen::Vector3d y;
    y << 1, 0, 1;
    Eigen::VectorXd lambdas(5);
    lambdas << 0, 0.5, 0.8, 0.9, 1;
    Eigen::MatrixXd out_beta_mat = logistic_elnet_coefs(X, y, lambdas, 1);
}

TEST(LogisticElnet, RidgeRuns)
{
    Eigen::Matrix3d X;
    X << -0.48, -1.33, 1.22,
        -0.90, 0.26, 0,
        1.39, 1.07, -1.22;
    Eigen::Vector3d y;
    y << 1, 0, 1;
    Eigen::VectorXd lambdas(6);
    lambdas << 0, 0.5, 0.8, 0.9, 1, 100;
    Eigen::MatrixXd out_beta_mat = logistic_elnet_coefs(X, y, lambdas, 0);
}

TEST(LogisticElnet, ElasticNetRuns)
{
    Eigen::Matrix3d X;
    X << -0.48, -1.33, 1.22,
        -0.90, 0.26, 0,
        1.39, 1.07, -1.22;
    Eigen::Vector3d y;
    y << 1, 0, 1;
    Eigen::VectorXd lambdas(5);
    lambdas << 0, 0.5, 0.8, 0.9, 1;
    Eigen::MatrixXd out_beta_mat = logistic_elnet_coefs(X, y, lambdas, 0.77);
}

TEST(LogisticElnet, EarlyStopping)
{
    Eigen::Matrix3d X;
    X << -0.48, -1.33, 1.22,
        -0.90, 0.26, 0,
        1.39, 1.07, -1.22;
    Eigen::Vector3d y;
    y << 1, 0, 1;
    Eigen::VectorXd lambdas(5);
    lambdas << 0, 0.5, 0.8, 0.9, 1;
    int expected_rows = 4;
    int expected_cols = 5;
    Eigen::MatrixXd out_beta_mat = logistic_elnet_coefs(X, y, lambdas, 0.77, false);
    int beta_rows = out_beta_mat.rows();
    int beta_cols = out_beta_mat.cols();
    EXPECT_EQ(beta_rows, expected_rows);
    EXPECT_EQ(beta_cols, expected_cols);
}
