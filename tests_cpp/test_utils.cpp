#include "gtest/gtest.h"
#include "utils.hpp"
#include "Eigen/Dense"

TEST(UtilsTests, TestGetSd)
{
    Eigen::VectorXd v(4);
    v << -1, 1, 1, -1;
    EXPECT_EQ(get_sd(v), 1);
}
