#ifndef LIBELNET_LINEAR_ELNET_H
#define LIBELNET_LINEAR_ELNET_H

#include "Eigen/Dense"

/** Optimization step for the linear lasso
 * 
 * @param X Predictors
 * @param y Response
 * @param lambda Regularization parameter
 * @param n_vars Number of variables / predictors
 * @param n_obs Number of observations / rows
 * @param init_beta Initial coefficients for warm starts
 * @param tol Tolerance level for stopping criterion
 * @param maxit Maximum number of iterations
 * @return Estimated lasso coefficients
*/
Eigen::VectorXd linear_lasso_optim(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &lambda,
    const int &n_vars,
    const int &n_obs,
    const Eigen::VectorXd &init_beta,
    const double &tol = 1e-7,
    const int &maxit = 1e+5);

/** Optimization step for the linear elastic net
 * 
 * @param X Predictors
 * @param y Response
 * @param lambda Regularization parameter
 * @param alpha Regularization hyperparameter
 * @param n_vars Number of variables / predictors
 * @param n_obs Number of observations / rows
 * @param init_beta Initial coefficients for warm starts
 * @param tol Tolerance level for stopping criterion
 * @param maxit Maximum number of iterations
 * @return Estimated elastic net coefficients
*/
Eigen::VectorXd linear_elastic_net_optim(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &lambda,
    const double &alpha,
    const int &n_vars,
    const int &n_obs,
    const Eigen::VectorXd &init_beta,
    const double &tol = 1e-7,
    const int &maxit = 1e+5);

/** Estimate matrix of coefficients for the linear lasso
 * 
 * @param X Predictors. Assumed to be standardized beforehand (mean 0 and variance 1)
 * @param y Response. Assumed to be standardized beforehand
 * @param lambdas Vector of regularization parameters. Assumed to be normalized 
 * via standard deviation of y beforehand
 * @param tol Tolerance level for stopping criterion
 * @param maxit Maximum number of iterations
 * @param devmax Stopping criterion for elastic net path with maximum deviance explained
 * @param fdev Stopping criterion for elastic net path with difference to previous deviance
 * @return Estimated coefficients for each lambda
*/
Eigen::MatrixXd linear_lasso_component(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &lambdas,
    const double &tol,
    const int &maxit,
    const double &devmax = 0.999,
    const double &fdev = 1e-5);

/** Estimate matrix of coefficients for the linear elastic net
 * 
 * @param X Predictors. Assumed to be standardized beforehand (mean 0 and variance 1)
 * @param y Response. Assumed to be standardized beforehand
 * @param lambdas Vector of regularization parameters. Assumed to be normalized 
 * via standard deviation of y beforehand
 * @param alpha Regularization hyper parameter
 * @param tol Tolerance level for stopping criterion
 * @param maxit Maximum number of iterations
 * @param devmax Stopping criterion for elastic net path with maximum deviance explained
 * @param fdev Stopping criterion for elastic net path with difference to previous deviance
 * @return Estimated coefficients for each lambda
*/
Eigen::MatrixXd linear_elastic_net_component(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &lambdas,
    const double &alpha,
    const double &tol,
    const int &maxit,
    const double &devmax = 0.999,
    const double &fdev = 1e-5);

/** Estimate elastic net for the linear model
 * 
 * @param X Predictors. Assumed to be standardized beforehand (mean 0 and variance 1)
 * @param y Response. Assumed to be standardized beforehand
 * @param lambdas Vector of regularization parameters. Assumed to be normalized 
 * via standard deviation of y beforehand
 * @param alpha Regularization hyperparameter. 1 for lasso,
 * 0 for ridge and (0, 1) for elastic net
 * @param tol Tolerance level for stopping criterion
 * @param maxit Maximum number of iterations
 * @return Estimated elastic net coefficients
*/
Eigen::MatrixXd linear_elnet_coefs(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &lambdas,
    const double &alpha,
    const double &tol = 1e-7,
    const int &maxit = 1e+5);

#endif
