#ifndef LIBELNET_UTILS_H
#define LIBELNET_UTILS_H

#include "Eigen/Dense"

struct StdOut
{
    Eigen::RowVectorXd X_means;
    Eigen::RowVectorXd X_stds;
    double y_mean;
    double y_std;
};

struct Coefs
{
    Eigen::VectorXd betas;
    double intercept;
};

struct WeightedInput
{
    Eigen::MatrixXd Xt;
    Eigen::VectorXd yt;
};

/** Standardize matrix of predictors X and response y inplace
 * 
 * @param X Matrix of predcitors X
 * @param y Response vector
 * @return X means, y mean, X standard deviations and y standard deviation
*/
StdOut standardize_inplace(Eigen::MatrixXd &X, Eigen::VectorXd &y);

/** Standardize beta coefficients
 * 
 * @param beta Beta coefficients
 * @param X_stds Predictors standard deviations
 * @param y_std Response standard deviation
 * @return Standardized coefficients
*/
Eigen::VectorXd standardize_beta(const Eigen::VectorXd &beta,
                                 const Eigen::VectorXd &X_stds,
                                 const double &y_std);

/** Destandardize coefficients
 * 
 * @param betas Vector of standardized coefficients
 * @param X_means Vector of predictors means
 * @param X_stds Vector of predictors standard deviations
 * @param y_mean Response mean
 * @param y_std Response standard deviation
 * @return Destandardized coefficients and intercept
*/
Coefs destandardize_coefs(const Eigen::VectorXd &betas,
                          const Eigen::VectorXd &X_means,
                          const Eigen::VectorXd &X_stds,
                          const double &y_mean,
                          const double &y_std);

/** Calculate null log-likelihood, i.e., the log-likelihood
 * for the benchmark (null) model with just an intercept
 * 
 * @param y Binary response vector
 * @return Null log-likelihood
*/
double get_null_loglik(const Eigen::VectorXd &y);

/** Calculate log-likelihood
 * 
 * @param probs Predicted probabilities by the model / algorithm
 * @param y Binary response vector
 * @param probs_eps Tolerance for the probabilities. Probabilities lower than
 * probs_eps will be set to probs_eps and probabilities higher than
 * 1 - probs_eps will be set to 1 - probs_eps
 * @return Log-likelihood
*/
double get_loglik(const Eigen::VectorXd &probs,
                  const Eigen::VectorXd &y,
                  const double &prob_eps = 1e-5);

/** Get probabilities for the model
 * 
 * @param X Predictors matrix
 * @param beta Beta (predictors) coefficients
 * @param intercept Intercept for the model
 * @return Predicted probabilities
*/
Eigen::VectorXd get_probs(const Eigen::MatrixXd &X,
                          const Eigen::VectorXd &beta,
                          const double &intercept,
                          const double &prob_eps = 1e-5);

#endif
