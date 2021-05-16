#ifndef LIBELNET_LOGISTIC_ELNET_H
#define LIBELNET_LOGISTIC_ELNET_H

#include "Eigen/Dense"

#include "utils.hpp"

/** Get weighted input for iteratively reweighted least squares (IRLS)
 * 
 * @param X Matrix of predictors
 * @param y Response vector
 * @param beta Vector of coefficients
 * @param intercept Intercept of logistic regression
 * @return Weighted input ready for IRLS
*/
WeightedInput get_weighted_inputs(const Eigen::MatrixXd &X,
                                  const Eigen::VectorXd &y,
                                  const Eigen::VectorXd &beta,
                                  const double &intercept);

/** Logistic elastic net optimization step
 * 
 * @param X Predictors matrix
 * @param response Response vector
 * @param lambdas Vector of regularization parameters
 * @param alpha Elastic net hyperparameter (0 = ridge, 1 = lasso, in between elastic net)
 * @param n_vars Number of variables / features / predictors
 * @param n_obs Number of observations
 * @param init_intercept Initial intercept
 * @param init_beta Initial beta coefficients
 * @param tol Internal stopping criterion for the coordinate descent loop
 * @param maxit Maximum number of iterations in the coordinate descent loop
 * @param tol_logit Internal stopping criterion for the IRLS 
 * (iteratively re-weighted least squares) step
 * @param maxit_logit Maximum number of iterations for the IRLS step 
*/
Coefs logistic_elnet_optim(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &lambda,
    const double &alpha,
    const int &n_vars,
    const int &n_obs,
    const double &init_intercept,
    const Eigen::VectorXd &init_beta,
    const double &tol = 1e-7,
    const int &maxit = 1e+5,
    const double &tol_logit = 1e-8,
    const int &maxit_logit = 25);

/** Estimate logistic elastic net coefficients
 * 
 * @param X Predictors matrix
 * @param response Response vector
 * @param lambdas Vector of regularization parameters
 * @param alpha Elastic net hyperparameter (0 = ridge, 1 = lasso, in between elastic net)
 * @param early_stopping Whether to stop the estimation along the lambdas if improvement
 * is not sufficient
 * @param tol Internal stopping criterion for the coordinate descent loop
 * @param maxit Maximum number of iterations in the coordinate descent loop
 * @return Matrix of coefficients for each lambda 
 * of shape (number of parameters + 1, number of lambdas)
 * + 1 because we include the intercept in the first row
*/
Eigen::MatrixXd logistic_elnet_coefs(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &lambdas,
    const double &alpha,
    const bool &early_stopping = true,
    const double &tol = 1e-7,
    const int &maxit = 1e+5,
    const double &devmax = 0.999,
    const double &fdev = 1e-5);

#endif
