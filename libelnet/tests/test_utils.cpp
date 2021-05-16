#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "utils.hpp"

TEST(Utils, Standardization)
{
    Eigen::MatrixXd X(3, 3);
    X << 1, 5, 3,
        2, 3, 1,
        8, 5, 2;
    Eigen::VectorXd y(3);
    y << 1, 2, 3;
    Eigen::MatrixXd expected_X(3, 3);
    expected_X << -0.862662, 0.707107, 1.22474,
        -0.539164, -1.41421, -1.22474,
        1.40183, 0.707107, 0;
    Eigen::VectorXd expected_y(3);
    expected_y << -1.22474, 0, 1.22474;
    Eigen::RowVectorXd expected_X_means(3);
    expected_X_means << 3.66667, 4.33333, 2;
    double expected_y_mean = 2;
    Eigen::RowVectorXd expected_X_stds(3);
    expected_X_stds << 3.09121, 0.942809, 0.816497;
    double expected_y_std = 0.816497;
    StdOut out = standardize_inplace(X, y);
    EXPECT_TRUE(X.isApprox(expected_X, 1e-5));
    EXPECT_TRUE(y.isApprox(expected_y, 1e-5));
    EXPECT_TRUE(out.X_means.isApprox(expected_X_means, 1e-5));
    EXPECT_NEAR(out.y_mean, expected_y_mean, 1e-5);
    EXPECT_TRUE(out.X_stds.isApprox(expected_X_stds, 1e-5));
    EXPECT_NEAR(out.y_std, expected_y_std, 1e-5);
}

TEST(Utils, CoefsStandardization)
{
    Eigen::VectorXd betas(3);
    betas << 1, 2, 3;
    Eigen::VectorXd X_stds(3);
    X_stds << 1, 2, 8;
    double y_std = 0.5;
    Eigen::VectorXd betas_std = standardize_beta(betas, X_stds, y_std);
    Eigen::VectorXd expected_betas_std(3);
    expected_betas_std << 2, 8, 48;
    EXPECT_EQ(betas_std, expected_betas_std);
}

TEST(Utils, CoefsDestandardization)
{
    Eigen::VectorXd betas(3);
    betas << 1, 2, 3;
    Eigen::VectorXd X_means(3);
    X_means << 5, 1, 2;
    Eigen::VectorXd X_stds(3);
    X_stds << 1, 2, 8;
    double y_mean = 1;
    double y_std = 0.5;
    Coefs out = destandardize_coefs(betas, X_means, X_stds, y_mean, y_std);
    Eigen::VectorXd expected_betas_destd(3);
    expected_betas_destd << 0.5, 0.5, 0.1875;
    const double expected_intercept = -2.375;
    EXPECT_EQ(out.betas, expected_betas_destd);
    EXPECT_EQ(out.intercept, expected_intercept);
}

TEST(Utils, NullLoglikelihood)
{
    Eigen::VectorXd y(5);
    y << 1, 0, 0, 1, 1;
    const double null_loglik = get_null_loglik(y);
    const double expected_null_loglik = -3.36506;
    EXPECT_NEAR(null_loglik, expected_null_loglik, 1e-5);
}

TEST(Utils, Loglikelihood)
{
    Eigen::VectorXd y(5);
    y << 1, 0, 0, 1, 1;
    Eigen::VectorXd probs(5);
    probs << 0.8, 0, 0.5, 0.3, 1;
    const double loglik = get_loglik(probs, y);
    const double expected_loglik = -2.12028;
    EXPECT_NEAR(loglik, expected_loglik, 1e-5);
}

TEST(Utils, GetProbs)
{
    Eigen::MatrixXd X(3, 3);
    X << 1, 5, 20,
        2, 3, 1,
        8, 5, 2;
    Eigen::VectorXd beta(3);
    beta << -1, .5, 1;
    const double intercept = -0.5;
    Eigen::VectorXd probs = get_probs(X, beta, intercept);
    Eigen::VectorXd expected_probs(3);
    expected_probs << 0.99999, 0.5, 0.0179862;
    EXPECT_TRUE(probs.isApprox(expected_probs, 1e-5));
}
