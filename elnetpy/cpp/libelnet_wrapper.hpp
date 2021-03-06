#ifndef LIBELNET_WRAPPER_H
#define LIBELNET_WRAPPER_H

#include "Eigen/Dense"
#include <pybind11/eigen.h>
#include "linear_elnet.hpp"

Eigen::MatrixXd linear_elnet(
    Eigen::Ref<Eigen::MatrixXd> X,
    Eigen::Ref<Eigen::VectorXd> y,
    Eigen::Ref<Eigen::VectorXd> lambdas,
    const double alpha,
    const bool early_stopping,
    const double tol = 1e-7,
    const double maxit = 1e+5);

#endif
