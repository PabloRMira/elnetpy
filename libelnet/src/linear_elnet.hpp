#ifndef LIBELNET_LINEAR_ELNET_H
#define LIBELNET_LINEAR_ELNET_H

#include "Eigen/Dense"

/** Optimization step for the linear elastic net
 * 
 * @param X Predictors
 * @param y Response
 * @param lambda Regularization parameter
 * @param n_vars Number of variables / predictors
 * @param n_obs Number of observations / rows
 * @param init_beta Initial coefficients for warm starts
 * @param init_beta_available Whether init_beta should be used as warm start,
 * defaults to false
 * @param tol Tolerance level for stopping criterion
 * @param maxit Maximum number of iterations
 * @return Estimated elastic net coefficients
*/
Eigen::VectorXd linear_elnet_optim(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &lambda,
    const int &n_vars,
    const int &n_obs,
    const Eigen::VectorXd &init_beta,
    const bool init_beta_available = false,
    const double &tol = 1e-7,
    const double &maxit = 1e+5);

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
    const double maxit = 1e+5);

#endif