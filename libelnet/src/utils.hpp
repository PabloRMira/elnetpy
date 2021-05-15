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

#endif
