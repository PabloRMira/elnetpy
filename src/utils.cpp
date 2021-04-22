#include <cmath>
#include "Eigen/Dense"
#include "utils.hpp"

double get_sd(const Eigen::VectorXd &v)
{
    /* Get standard deviation of v */
    Eigen::VectorXd sqs = v.array().pow(2);
    double mean_sqs = sqs.mean();
    double sd = std::sqrt(mean_sqs);
    return sd;
}
