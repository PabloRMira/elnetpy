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

double get_null_loglik(const Eigen::VectorXd &y)
{
    const double y_mean = y.array().mean();
    const int n1 = y.array().sum();
    const int n0 = y.size() - n1;
    const double null_loglik = (n1 * std::log(y_mean)) + (n0 * std::log(1 - y_mean));
    return null_loglik;
}

double get_loglik(const Eigen::VectorXd &probs,
                  const Eigen::VectorXd &y,
                  const double &prob_eps)
{
    const double min_prob = prob_eps;
    const double max_prob = 1 - prob_eps;
    Eigen::VectorXd probs_adj = probs;
    probs_adj = (probs_adj.array() > max_prob).select(max_prob, probs_adj);
    probs_adj = (probs_adj.array() < min_prob).select(min_prob, probs_adj);
    const double loglik = ((y.array() * probs_adj.array().log()) +
                           ((1 - y.array()) * (1 - probs_adj.array()).log()))
                              .sum();
    return loglik;
}

Eigen::VectorXd get_probs(const Eigen::MatrixXd &X,
                          const Eigen::VectorXd &beta,
                          const double &intercept,
                          const double &prob_eps)
{
    const double min_prob = prob_eps;
    const double max_prob = 1 - prob_eps;
    const Eigen::VectorXd neg_linear_effects = -(intercept + (X * beta).array());
    Eigen::VectorXd probs = (1 + neg_linear_effects.array().exp()).cwiseInverse();
    probs = (probs.array() > max_prob).select(max_prob, probs);
    probs = (probs.array() < min_prob).select(min_prob, probs);
    return probs;
}
