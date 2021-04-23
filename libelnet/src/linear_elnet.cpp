#include <cmath>
#include <iostream>
#include "Eigen/Dense"
#include "linear_elnet.hpp"

Eigen::VectorXd linear_elnet_optim(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &lambda,
    const int &n_vars,
    const int &n_obs,
    const Eigen::VectorXd &init_beta,
    const bool init_beta_available = false,
    const double &tol = 1e-7,
    const double &maxit = 1e+5)
{
    Eigen::VectorXd previous_beta;
    if (init_beta_available) /* if warm start available */
    {
        previous_beta = init_beta;
    }
    else
    {
        previous_beta = Eigen::VectorXd::Zero(n_vars);
    }
    Eigen::VectorXd beta = previous_beta;
    int iter = 0;
    double theta;
    double theta_abs;
    int theta_sign;
    double diff_norm = 1;
    while ((diff_norm > tol) && (iter < maxit))
    {
        for (int j = 0; j < n_vars; j++)
        {
            theta = ((y - X * beta).cwiseProduct(X.col(j))).mean() + beta(j);
            beta(j) = 0.0;
            theta_abs = std::abs(theta);
            if (theta_abs > lambda)
            {
                theta_sign = (theta > 0) ? 1 : -1;
                beta(j) = theta_sign * (theta_abs - lambda);
            }
        }
        diff_norm = (beta - previous_beta).array().pow(2).maxCoeff();
        iter += 1;
        previous_beta = beta;
    }
    /* and yet again a last round like J. Friedman's Fortran code for glmnet */
    for (int j = 0; j < n_vars; j++)
    {
        theta = ((y - X * beta).cwiseProduct(X.col(j))).mean() + beta(j);
        beta(j) = 0.0;
        theta_abs = std::abs(theta);
        if (theta_abs > lambda)
        {
            theta_sign = (theta > 0) ? 1 : -1;
            beta(j) = theta_sign * (theta_abs - lambda);
        }
    }
    iter += 1;
    if (iter >= maxit)
    {
        std::cout << "Maximum iteration achieved" << std::endl;
    }
    return beta;
}

/** Estimate elastic net for the linear model
 * 
 * @param X Predictors. Assumed to be standardized beforehand (mean 0 and variance 1)
 * @param y Response. Assumed to be standardized beforehand
 * @param lambdas Vector of regularization parameters. Assumed to be normalized 
 * via standard deviation of y beforehand
 * @param tol Tolerance level for stopping criterion
 * @param maxit Maximum number of iterations
 * @return Estimated elastic net coefficients
*/
Eigen::MatrixXd linear_elnet_coefs(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &lambdas,
    const double tol = 1e-7,
    const double maxit = 1e+5)
{
    const int n_obs = X.rows();
    const int n_vars = X.cols();
    const int n_lambdas = lambdas.size();
    Eigen::MatrixXd beta_mat = Eigen::MatrixXd::Zero(n_vars, n_lambdas);
    Eigen::VectorXd pseudo_init_beta = Eigen::VectorXd::Zero(1);
    beta_mat.col(0) = linear_elnet_optim(X, y, lambdas(0),
                                         n_vars, n_obs,
                                         pseudo_init_beta, false,
                                         tol, maxit);
    if (n_lambdas > 1)
    {
        for (int k = 1; k < n_lambdas; k++)
        {
            beta_mat.col(k) = linear_elnet_optim(X, y, lambdas(0),
                                                 n_vars, n_obs,
                                                 beta_mat.col(k - 1), true,
                                                 tol, maxit);
        }
    }
    return beta_mat;
}
