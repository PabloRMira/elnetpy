#include <cmath>
#include <iostream>
#include "Eigen/Dense"
#include "linear_elnet.hpp"

Eigen::VectorXd linear_elastic_net_optim(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &lambda,
    const double &alpha,
    const int &n_vars,
    const int &n_obs,
    const Eigen::VectorXd &init_beta,
    const double &tol,
    const int &maxit)
{
    Eigen::VectorXd previous_beta = init_beta; /* warm start */
    Eigen::VectorXd beta = previous_beta;      /* initialize with previous beta */
    int iter = 0;
    double theta;
    double theta_abs;
    int theta_sign;
    double diff_norm = 1;
    double lambda_alpha;
    double denomi; /* denominator for update */
    lambda_alpha = lambda * alpha;
    denomi = 1 + (lambda * (1 - alpha));
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

Eigen::MatrixXd linear_elastic_net_component(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &lambdas,
    const double &alpha,
    const double &tol,
    const int &maxit,
    const double &devmax,
    const double &fdev)
{
    const int n_obs = X.rows();
    const int n_vars = X.cols();
    const int n_lambdas = lambdas.size();
    double lambda = lambdas(0);
    double mse;          /* mse = mean squared error = deviance (for gaussian model like this) */
    double rsq;          /* R squared = this is called deviance explained in glmnet */
    double rsq_prev = 0; /* previous R-squared / deviance, for the first (intercept only) is 0 */
    double rsq_change;   /* change in deviance for stopping criterion */
    double fdev_crit;    /* adjusted stopping criterion for more efficiency */
    Eigen::VectorXd betas;
    Eigen::MatrixXd beta_mat = Eigen::MatrixXd::Zero(n_vars, n_lambdas);
    Eigen::VectorXd init_beta;
    if (alpha == 0)
    {
        /* for the ridge estimator glmnet assumes 0 vector with highest lambda */
        beta_mat.col(0) = Eigen::VectorXd::Zero(n_vars);
    }
    else
    {
        init_beta = Eigen::VectorXd::Zero(n_vars);
        beta_mat.col(0) = linear_elastic_net_optim(X, y, lambda, alpha,
                                                   n_vars, n_obs,
                                                   init_beta,
                                                   tol, maxit);
    }
    if (n_lambdas > 1)
    {
        for (int k = 1; k < n_lambdas; k++)
        {
            lambda = lambdas(k);
            init_beta = beta_mat.col(k - 1);
            betas = linear_elastic_net_optim(X, y, lambda, alpha,
                                             n_vars, n_obs,
                                             init_beta,
                                             tol, maxit);
            beta_mat.col(k) = betas;
            mse = (y - X * betas).array().square().mean();
            /* R squared is actually 1 - ( mse(model) / mse(only intercept) )
               But because of standardization of y, mse(only intercept) = 1 and
               this simplifies the calculation as below */
            rsq = 1 - mse;
            rsq_change = rsq - rsq_prev;
            fdev_crit = fdev * rsq;
            if (rsq > devmax || rsq_change < fdev_crit)
            {
                beta_mat = beta_mat.leftCols(k + 1);
                break;
            }
            rsq_prev = rsq;
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
    beta_mat = linear_elastic_net_component(X, y, lambdas, alpha, tol, maxit);
    return beta_mat;
}
