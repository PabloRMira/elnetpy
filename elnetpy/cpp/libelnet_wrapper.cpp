#include "../../libelnet/lib/eigen/Eigen/Dense"
#include <pybind11/eigen.h>
#include "../../libelnet/src/linear_elnet.hpp"
#include "libelnet_wrapper.hpp"

Eigen::MatrixXd linear_elnet(
    Eigen::Ref<Eigen::MatrixXd> X,
    Eigen::Ref<Eigen::VectorXd> y,
    Eigen::Ref<Eigen::VectorXd> lambdas,
    const double tol,
    const double maxit)
{
    Eigen::MatrixXd beta_mat = linear_elnet_coefs(X, y, lambdas, tol, maxit);
    return beta_mat;
}
