#include "Eigen/Dense"
#include <cmath>
#include "utils.hpp"
#include "logistic_elnet.hpp"

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

Eigen::VectorXd standardize_beta(const Eigen::VectorXd &beta,
                                 const Eigen::VectorXd &X_stds,
                                 const double &y_std)
{
    Eigen::VectorXd betas_std = beta.array() / y_std;
    betas_std = betas_std.array() * X_stds.array();
    return betas_std;
}

Coefs destandardize_coefs(const Eigen::VectorXd &betas,
                          const Eigen::VectorXd &X_means,
                          const Eigen::VectorXd &X_stds,
                          const double &y_mean,
                          const double &y_std)
{
    Coefs out;
    Eigen::VectorXd beta_destd = betas.array() * y_std;
    out.betas = beta_destd.array() / X_stds.array();
    out.intercept = y_mean - out.betas.dot(X_means);
    return out;
}
