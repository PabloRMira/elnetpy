#include "Eigen/Dense"
#include <cmath>
#include "utils.hpp"

StdOut standardize_inplace(Eigen::MatrixXd &X, Eigen::VectorXd &y)
{
    StdOut out;
    out.X_means = X.array().colwise().mean();
    out.y_mean = y.array().mean();
    X = X.array().rowwise() - out.X_means.array();
    y = y.array() - out.y_mean;
    out.X_stds = X.array().pow(2).colwise().mean().array().sqrt();
    out.y_std = std::sqrt(y.array().pow(2).mean());
    X = X.array().rowwise() / out.X_stds.array();
    y = y.array() / out.y_std;
    return out;
}
