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
                                 const Eigen::VectorXd X_stds,
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

#endif
