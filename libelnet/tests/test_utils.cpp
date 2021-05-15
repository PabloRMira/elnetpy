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
    Eigen::VectorXd expected_X_means(3);
    expected_X_means << 3.66667, 4.33333, 2;
    double expected_y_mean = 2;
    Eigen::VectorXd expected_X_stds(3);
    expected_X_stds << 3.09121, 0.942809, 0.816497;
    double expected_y_std = 0.816497;
    StdOut out = standardize_inplace(X, y);
    EXPECT_EQ(X, expected_X);
    EXPECT_EQ(y, expected_y);
    EXPECT_EQ(out.X_means, expected_X_means);
    EXPECT_EQ(out.y_mean, expected_y_mean);
    EXPECT_EQ(out.X_stds, expected_X_stds);
    EXPECT_EQ(out.y_std, expected_y_std);
}
