#include <cmath>
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/LU"
#include "linear_elnet.hpp"

Eigen::VectorXd linear_lasso_optim(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &lambda,
    const int &n_vars,
    const int &n_obs,
    const Eigen::VectorXd &init_beta,
    const bool init_beta_available,
    const double &tol,
    const int &maxit)
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

Eigen::VectorXd linear_elastic_net_optim(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &lambda,
    const double &alpha,
    const int &n_vars,
    const int &n_obs,
    const Eigen::VectorXd &init_beta,
    const bool init_beta_available,
    const double &tol,
    const int &maxit)
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
    double lambda_alpha = lambda * alpha;
    /* denominator for update */
    double denomi = 1 + (lambda * (1 - alpha));
    while ((diff_norm > tol) && (iter < maxit))
    {
        for (int j = 0; j < n_vars; j++)
        {
            theta = ((y - X * beta).cwiseProduct(X.col(j))).mean() + beta(j);
            beta(j) = 0.0;
            theta_abs = std::abs(theta);
            if (theta_abs > lambda_alpha)
            {
                theta_sign = (theta > 0) ? 1 : -1;
                beta(j) = (theta_sign * (theta_abs - lambda_alpha)) / denomi;
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
        if (theta_abs > lambda_alpha)
        {
            theta_sign = (theta > 0) ? 1 : -1;
            beta(j) = (theta_sign * (theta_abs - lambda_alpha)) / denomi;
        }
    }
    iter += 1;
    if (iter >= maxit)
    {
        std::cout << "Maximum iteration achieved" << std::endl;
    }
    return beta;
}

Eigen::MatrixXd linear_ridge_optim(
    const Eigen::MatrixXd &X,
    const Eigen::MatrixXd &y,
    const double &lambda,
    const int &n_vars,
    const int &n_obs)
{
    Eigen::MatrixXd regul = Eigen::MatrixXd::Identity(n_obs, n_vars);
    regul.array() *= lambda;
    Eigen::MatrixXd Xt = X.transpose();
    Eigen::MatrixXd XX_ridge = (Xt * X) + regul;
    Eigen::VectorXd Xy = Xt * y;
    Eigen::VectorXd ridge_est = XX_ridge.lu().solve(Xy);
    return ridge_est;
}

Eigen::MatrixXd linear_lasso_component(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &lambdas,
    const double &tol,
    const int &maxit)
{
    const int n_obs = X.rows();
    const int n_vars = X.cols();
    const int n_lambdas = lambdas.size();
    double lambda = lambdas(0);
    Eigen::MatrixXd beta_mat = Eigen::MatrixXd::Zero(n_vars, n_lambdas);
    Eigen::VectorXd init_beta = Eigen::VectorXd::Zero(1);
    beta_mat.col(0) = linear_lasso_optim(X, y, lambda,
                                         n_vars, n_obs,
                                         init_beta, false,
                                         tol, maxit);
    if (n_lambdas > 1)
    {
        for (int k = 1; k < n_lambdas; k++)
        {
            lambda = lambdas(k);
            init_beta = beta_mat.col(k - 1);
            beta_mat.col(k) = linear_lasso_optim(X, y, lambda,
                                                 n_vars, n_obs,
                                                 init_beta, true,
                                                 tol, maxit);
        }
    }
    return beta_mat;
}

Eigen::MatrixXd linear_elastic_net_component(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &lambdas,
    const double &alpha,
    const double &tol,
    const int &maxit)
{
    const int n_obs = X.rows();
    const int n_vars = X.cols();
    const int n_lambdas = lambdas.size();
    double lambda = lambdas(0);
    Eigen::MatrixXd beta_mat = Eigen::MatrixXd::Zero(n_vars, n_lambdas);
    Eigen::VectorXd init_beta = Eigen::VectorXd::Zero(1);
    beta_mat.col(0) = linear_elastic_net_optim(X, y, lambda, alpha,
                                               n_vars, n_obs,
                                               init_beta, false,
                                               tol, maxit);
    if (n_lambdas > 1)
    {
        for (int k = 1; k < n_lambdas; k++)
        {
            lambda = lambdas(k);
            init_beta = beta_mat.col(k - 1);
            beta_mat.col(k) = linear_elastic_net_optim(X, y, lambda, alpha,
                                                       n_vars, n_obs,
                                                       init_beta, true,
                                                       tol, maxit);
        }
    }
    return beta_mat;
}

Eigen::MatrixXd linear_ridge_component(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &lambdas)
{
    const int n_obs = X.rows();
    const int n_vars = X.cols();
    const int n_lambdas = lambdas.size();
    double lambda = lambdas(0);
    Eigen::MatrixXd beta_mat = Eigen::MatrixXd::Zero(n_vars, n_lambdas);
    Eigen::VectorXd init_beta = Eigen::VectorXd::Zero(1);
    beta_mat.col(0) = linear_ridge_optim(X, y, lambda, n_obs, n_vars);
    if (n_lambdas > 1)
    {
        for (int k = 1; k < n_lambdas; k++)
        {
            lambda = lambdas(k);
            beta_mat.col(k) = linear_ridge_optim(X, y, lambda, n_vars, n_obs);
        }
    }
    return beta_mat;
}

Eigen::MatrixXd linear_elnet_coefs(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &lambdas,
    const double &alpha,
    const double &tol,
    const int &maxit)
{
    Eigen::MatrixXd beta_mat;
    if (alpha == 1)
    {
        beta_mat = linear_lasso_component(X, y, lambdas, tol, maxit);
    }
    else if (alpha == 0)
    {
        beta_mat = linear_ridge_component(X, y, lambdas);
    }
    else
    {
        beta_mat = linear_elastic_net_component(X, y, lambdas, alpha, tol, maxit);
    }
    return beta_mat;
}
