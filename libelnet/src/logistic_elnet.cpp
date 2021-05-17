#include <cmath>
#include <iostream>
#include "Eigen/Dense"
#include "linear_elnet.hpp"
#include "logistic_elnet.hpp"
#include "utils.hpp"

WeightedInput get_weighted_inputs(const Eigen::MatrixXd &X,
                                  const Eigen::VectorXd &y,
                                  const Eigen::VectorXd &beta,
                                  const double &intercept)
{
    const Eigen::VectorXd probs = get_probs(X, beta, intercept);
    const Eigen::VectorXd weights_sqrt = (probs.array() * (1 - probs.array())).sqrt();
    const Eigen::VectorXd weights_invsqrt = weights_sqrt.array().inverse();
    WeightedInput out;
    out.Xt = X.array().colwise() * weights_sqrt.array();
    out.yt = ((intercept + (X * beta).array()) * weights_sqrt.array()) + ((y - probs).array() * weights_invsqrt.array());
    return out;
}

Coefs logistic_elnet_optim(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &lambda,
    const double &alpha,
    const int &n_vars,
    const int &n_obs,
    const double &init_intercept,
    const Eigen::VectorXd &init_beta,
    const double &tol,
    const int &maxit,
    const double &tol_logit,
    const int &maxit_logit)
{
    Eigen::VectorXd previous_beta = init_beta; /* warm start */
    Eigen::VectorXd beta = previous_beta;      /* initialize with previous beta */
    double intercept = init_intercept;
    double diff_norm = 1;
    int iter = 0;
    WeightedInput w_input;
    StdOut std_out;
    double lambda_std;
    Coefs destd_coefs;
    while ((diff_norm > tol_logit) && (iter < maxit_logit))
    {
        w_input = get_weighted_inputs(X, y, beta, intercept);
        std_out = standardize_inplace(w_input.Xt, w_input.yt);
        lambda_std = lambda / std_out.y_std;
        /* Compare to previous_beta in beta's scale */
        previous_beta = standardize_beta(previous_beta, std_out.X_stds, std_out.y_std);
        beta = linear_elastic_net_optim(w_input.Xt, w_input.yt,
                                        lambda_std, alpha,
                                        n_vars, n_obs,
                                        previous_beta, tol, maxit);
        diff_norm = (beta - previous_beta).array().pow(2).maxCoeff();
        /* Destandardize coefficients for previous_beta to be compared on the right scale */
        destd_coefs = destandardize_coefs(beta,
                                          std_out.X_means, std_out.X_stds,
                                          std_out.y_mean, std_out.y_std);
        previous_beta = destd_coefs.betas; /* Update beta */
        intercept = destd_coefs.intercept; /* Update intercept */
        iter += 1;
    }
    if (iter >= maxit_logit)
    {
        std::cout << "Maximum iteration in logit step achieved" << std::endl;
    }
    return destd_coefs;
}

Eigen::MatrixXd logistic_elnet_coefs(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &lambdas,
    const double &alpha,
    const bool &early_stopping,
    const double &tol,
    const int &maxit,
    const double &devmax,
    const double &fdev)
{
    const int n_obs = X.rows();
    const int n_vars = X.cols();
    const int n_lambdas = lambdas.size();
    const double null_loglik = get_null_loglik(y);
    double loglik;
    double lambda = lambdas(0);
    Eigen::VectorXd probs;
    double rsq;          /* R squared = this is called deviance explained in glmnet */
    double rsq_prev = 0; /* previous R-squared / deviance, for the first (intercept only) is 0 */
    double rsq_change;   /* change in deviance for stopping criterion */
    double fdev_crit;    /* adjusted stopping criterion for more efficiency */
    Coefs coefs;
    Eigen::VectorXd intercepts = Eigen::RowVectorXd::Zero(n_lambdas);
    Eigen::MatrixXd beta_mat = Eigen::MatrixXd::Zero(n_vars, n_lambdas);
    Eigen::MatrixXd beta_mat_full = Eigen::MatrixXd::Zero(n_vars + 1, n_lambdas);
    Eigen::VectorXd init_beta;
    double init_intercept;
    if (alpha == 0)
    {
        /* for the ridge estimator glmnet assumes 0 vector with highest lambda */
        beta_mat.col(0) = Eigen::VectorXd::Zero(n_vars);
    }
    else
    {
        init_intercept = y.array().mean();
        init_beta = Eigen::VectorXd::Zero(n_vars);
        coefs = logistic_elnet_optim(X, y, lambda, alpha,
                                     n_vars, n_obs,
                                     init_intercept, init_beta,
                                     tol, maxit);
        beta_mat.col(0) = coefs.betas;
        intercepts(0) = coefs.intercept;
    }
    if (n_lambdas > 1)
    {
        for (int k = 1; k < n_lambdas; k++)
        {
            lambda = lambdas(k);
            init_beta = beta_mat.col(k - 1);
            init_intercept = intercepts(k - 1);
            coefs = logistic_elnet_optim(X, y, lambda, alpha,
                                         n_vars, n_obs,
                                         init_intercept, init_beta,
                                         tol, maxit);
            beta_mat.col(k) = coefs.betas;
            intercepts(k) = coefs.intercept;
            if (early_stopping)
            {
                probs = get_probs(X, coefs.betas, coefs.intercept);
                loglik = get_loglik(probs, y);
                rsq = 1 - (loglik / null_loglik);
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
    }
    beta_mat_full.topRows(1) = intercepts;
    beta_mat_full.middleRows(1, n_vars) = beta_mat;
    return beta_mat_full;
}
