"Linear Elastic Net"

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

from elnetpy import linear_elnet
from elnetpy.utils import (
    standardize_inputs,
    destandardize_coefs,
    get_lambda_path,
    get_linear_scorer,
    score_multiple,
    check_fitted_model,
    interpolate_model,
)


class Elnet(BaseEstimator):
    """Elastic Net estimator for linear models"""

    def __init__(
        self,
        alpha=1,
        lambdas=None,
        n_lambda=100,
        min_lambda_ratio=1e-4,
        n_splits=1,
        scoring="mean_squared_error",
        n_jobs=1,
        tol=1e-7,
        max_iter=1e5,
        random_state=182,
    ):
        self._validate_input(
            alpha,
            lambdas,
            n_lambda,
            min_lambda_ratio,
            n_jobs,
            tol,
            max_iter,
            n_splits,
            scoring,
            random_state,
        )
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambda = n_lambda
        self.min_lambda_ratio = min_lambda_ratio
        self.n_splits = n_splits
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):

        # for matrices to be passed by reference with pybind11
        # we need Fortran column major storage order
        _X = X.astype(dtype="float64", order="F", copy=True)
        _y = y.astype(dtype="float64", copy=True)

        # fit on the whole dataset
        self._fit(_X, _y)

        # cross validation
        if self.n_splits > 1:

            self._fit_cv(_X, _y)

        return self

    def _fit(self, X, y):

        X, y, X_means, X_stds, y_mean, y_std = standardize_inputs(
            X, y, return_means_stds=True
        )

        if self.lambdas is None:
            lambda_path_std = get_lambda_path(
                X, y, self.alpha, self.min_lambda_ratio, self.n_lambda
            )
            self.lambda_path_ = lambda_path_std * y_std
        else:
            lambda_path_std = (
                self.lambdas
                if isinstance(self.lambdas, np.ndarray)
                else np.array([self.lambdas])
            ).astype(dtype="float64")
            self.lambda_path_ = lambda_path_std
            lambda_path_std /= y_std

        # get standardized coefficients
        early_stopping = True
        coefs_mat = linear_elnet(
            X, y, lambda_path_std, self.alpha, early_stopping, self.tol, self.max_iter
        )

        # destandardize coefficients and get intercepts
        self.coef_path_, self.intercept_path_ = destandardize_coefs(
            coefs_mat, X_means, X_stds, y_mean, y_std
        )

        # truncate lambda sequence for stopping criterion
        self.lambda_path_ = self.lambda_path_[0 : self.coef_path_.shape[1]]

        return self

    def _fit_cv(self, X, y):

        splits = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        ).split(X)

        scorer_dict = get_linear_scorer[self.scoring]
        scorer = scorer_dict["scorer"]
        greater_better = scorer_dict["greater_better"]

        scores = Parallel(n_jobs=self.n_splits, prefer="threads")(
            delayed(_score_one)(
                X,
                y,
                self.lambda_path_,
                self.alpha,
                train_idx,
                test_idx,
                scorer,
                self.tol,
                self.max_iter,
            )
            for (train_idx, test_idx) in splits
        )
        scores = np.array(scores) if greater_better else -np.array(scores)
        self.cv_mean = scores.mean(axis=0)
        self.cv_std = scores.std(axis=0)
        self.cv_up = self.cv_mean + self.cv_std
        self.cv_low = self.cv_mean - self.cv_std

        lambda_best_idx = np.argmax(self.cv_mean)
        self.lambda_best_ = self.lambda_path_[lambda_best_idx]

        lambda_1std_idx = int(np.argwhere(self.cv_up >= self.cv_mean)[0])
        self.lambda_1std_ = self.lambda_path_[lambda_1std_idx]

        self.coef_ = self.coef_path_[:, lambda_1std_idx]
        self.intercept_ = self.intercept_path_[lambda_1std_idx]

        return self

    def predict(self, X, lamb=None):

        check_fitted_model(self)

        if lamb is None and not hasattr(self, "lambda_1std_"):
            raise Exception(
                """Please provide a lamb value for predict
                   or fit the estimator via cross validation
                   to use lambda_1std as default"""
            )
        elif lamb is None and hasattr(self, "lambda_1std_"):
            coef = self.coef_
            intercept = self.intercept_
        else:
            coef, intercept = interpolate_model(
                lamb, self.lambda_path, self.coef_path_, self.intercept_path_
            )

        y_pred = X.dot(coef) + intercept

        if y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze()

        return y_pred

    @staticmethod
    def _validate_input(
        alpha,
        lambdas,
        n_lambda,
        min_lambda_ratio,
        n_jobs,
        tol,
        max_iter,
        n_splits,
        scoring,
        random_state,
    ):
        if alpha < 0 or alpha > 1:
            raise ValueError(
                f"""Alpha should be between 0 and 1 inclusively.
                Your input alpha is {alpha}"""
            )
        if isinstance(lambdas, str):
            raise ValueError("lambdas should be either None, numeric or numpy array")
        if n_lambda <= 5:
            raise ValueError("n_lambda should be greater than 5")
        if min_lambda_ratio <= 0 or min_lambda_ratio >= 1:
            raise ValueError(
                f"""min_lambda_ratio should be between 0 and 1 exclusively.
                You input min_lambda_ratio is {min_lambda_ratio}"""
            )
        if not isinstance(n_jobs, int):
            raise ValueError(
                f"""n_jobs should be integer valued.
                Your input n_jobs is {n_jobs}"""
            )
        if n_jobs < -1 or n_jobs == 0:
            raise ValueError(
                f"""n_jobs should be greater than -1 and not 0.
                Your input n_jobs is {n_jobs}"""
            )
        if tol <= 0 or tol >= 1:
            raise ValueError(
                f"""tol should be between 0 and 1 exclusively.
                Your input tol is {tol}"""
            )
        if max_iter < 100:
            raise ValueError(
                f"""max_iter should be greater than or equal 100
                Your input max_iter is {max_iter}"""
            )
        if n_splits <= 0 or n_splits >= 100:
            raise ValueError(
                f"""n_splits should be between 1 and 100
                Your input n_splits is {n_splits}"""
            )
        if not isinstance(scoring, str) or scoring not in get_linear_scorer.keys():
            raise ValueError(
                f"""Only {", ".join(get_linear_scorer.keys())} are implemented
                Your input scoring is {scoring}"""
            )
        if not isinstance(random_state, int):
            raise ValueError("random_state should be an integer")


def _score_one(X, y, lambda_path, alpha, train_idx, test_idx, scorer, tol, max_iter):

    X_train = X[train_idx, :].astype(dtype="float64", order="F")
    y_train = y[train_idx]

    X_test = X[test_idx, :]
    y_test = y[test_idx]

    X_train, y_train, X_means, X_stds, y_mean, y_std = standardize_inputs(
        X_train, y_train, return_means_stds=True
    )

    # standardize lambdas
    lambda_path_std = lambda_path / y_std

    # get standardized coefficients
    coefs_mat = linear_elnet(
        X_train, y_train, lambda_path_std, alpha, False, tol, max_iter
    )
    coefs_mat, intercepts = destandardize_coefs(
        coefs_mat, X_means, X_stds, y_mean, y_std
    )

    y_preds_mat = X_test.dot(coefs_mat) + intercepts
    scores = score_multiple(y_test, y_preds_mat, scorer)
    return scores
