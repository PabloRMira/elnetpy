#ifndef LIBELNET_WRAPPER_H
#define LIBELNET_WRAPPER_H

#include "../../libelnet/lib/eigen/Eigen/Dense"
#include <pybind11/eigen.h>
#include "../../libelnet/src/linear_elnet.hpp"

Eigen::MatrixXd linear_elnet(
    Eigen::Ref<Eigen::MatrixXd> X,
    Eigen::Ref<Eigen::VectorXd> y,
    Eigen::Ref<Eigen::VectorXd> lambdas,
    const double tol = 1e-7,
    const double maxit = 1e+5);

#endif
