import logging

import numpy as np
import downhill
import theano
import theano.tensor as TT
import theano.sparse as tsparse
import sklearn.base as base
import sklearn.neighbors as neighbors
import sklearn.cross_validation as cv
import scipy.sparse as sparse

import util


class XYLDataset(object):
    """ Dataset object for downhill, (callable) because, for each batch,
    L is the same, but subsets of X and y vary"""

    def __init__(self, X, Y, L, batch_size=64):
        self.X = X
        self.Y = Y
        self.L = L
        self.batch_size = batch_size

    def __call__(self):
        batch = np.random.random_integers(0, self.X.shape[0] - 1, (self.batch_size,))
        return self.X[batch], self.Y[batch], self.L


class XYDataset(object):
    def __init__(self, X, Y, batch_size=64):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size

    def __call__(self):
        batch = np.random.random_integers(0, self.X.shape[0] - 1, (self.batch_size,))
        return self.X[batch], self.Y[batch]


class PathRegularizer(object):
    def test_regularization_path(self, X, y, X_valid, y_valid):
        # think there's an actual proper way to do this... ignore
        lambdas = [900, 600, 300, 100, 30, 10, 3, 1]
        for _lambda in lambdas:
            self.lambda_1 = _lambda
            self.fit(X, y)
            preds = self.predict(X_valid)
            score = np.corrcoef([y_valid[:,0], preds[:,0]])[0, 1]
            n_active_features = (self.w.get_value() > 0.0001).sum()
            logging.info(
                "lambda_1=%s,n_active_features=%s,corr=%s",
                _lambda,
                n_active_features,
                score
            )
            yield _lambda, n_active_features, score


class SparseRegressor(
        base.BaseEstimator, base.RegressorMixin, PathRegularizer
):
    def __init__(
            self,
            lambda_1=1,
            lambda_2=1,
            batch_size=32,
            default_downhill_args=dict(
                train_batches=128,
                valid_batches=1,
                patience=4,
                validate_every=4,
                monitor_gradients=False,
            ),
            downhill_args=dict(),
    ):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.batch_size = batch_size
        self.downhill_args = {k: v for k, v in default_downhill_args.items()}
        self.downhill_args.update(downhill_args)

    def fit(self, X, y):
        self.w = theano.shared(
            value=np.random.normal(0, 0.001, (X.shape[1], 1)),  # random initialize
            name="w",
            borrow=False
        )

        x_ = TT.matrix("X")
        y_ = TT.matrix("y")

        e = ((y_ - TT.dot(x_, self.w)) ** 2).sum()
        l1_penalty = abs(self.w).sum()
        l2_penalty = TT.sqrt((self.w * self.w).sum())

        loss = (
            e +
            self.lambda_1 * l1_penalty +
            self.lambda_2 * l2_penalty
        ).sum()

        x_train, x_valid, y_train, y_valid = cv.train_test_split(X, y)

        downhill.minimize(
            loss,
            XYDataset(x_train, y_train, batch_size=self.batch_size),
            valid=XYDataset(x_valid, y_valid, batch_size=x_valid.shape[0]),
            params=[self.w],
            inputs=[x_, y_],
            algo="rmsprop",
            **self.downhill_args
        )
        w = self.w.get_value()
        self.coef_dist = [
            (abs(w) > x).sum() for x in [0.01, 0.001, 0.0001, 0.00001, 0.000001]
        ]

    def predict(self, X):
        return np.dot(X, self.w.get_value())


class SSparseRegressor(
        base.BaseEstimator, base.RegressorMixin, PathRegularizer
):
    def __init__(
            self,
            to_sparsify_on,
            k_neighbors=10,
            alpha=1,
            lambda_1=1,
            lambda_2=1,
            batch_size=32,
            default_downhill_args=dict(
                train_batches=128,
                valid_batches=1,
                patience=4,
                validate_every=4,
                monitor_gradients=False,
            ),
            downhill_args=dict(),
            select_cols=None
    ):
        if select_cols is not None:
            self.select_cols = np.array(select_cols)
        else:
            self.select_cols = None

        # these are the feature regressions the Laplacian will be built from
        if self.select_cols is not None:
            _to_sparsify_on = to_sparsify_on[self.select_cols, :]
        else:
            _to_sparsify_on = to_sparsify_on

        kn = neighbors.kneighbors_graph(
            _to_sparsify_on,
            k_neighbors,
            mode="distance",
            metric="euclidean"
        )
        self.L = sparse.csgraph.laplacian(
            util.symmetrify_similarity(kn, _to_sparsify_on)
        )
        self.alpha = alpha
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        self.batch_size = batch_size
        self.downhill_args = {k: v for k, v in default_downhill_args.items()}
        self.downhill_args.update(downhill_args)

    def fit(self, X, y):
        if self.select_cols is not None:
            _X = X[:, self.select_cols]
        else:
            _X = X

        self.w = theano.shared(
            value=np.random.normal(0, 0.001, (_X.shape[1], 1)),  # random initialize
            name="w",
            borrow=False
        )


        x_ = TT.matrix("X")
        y_ = TT.matrix("y")
        l_ = tsparse.csr_matrix("l")

        e = ((y_ - TT.dot(x_, self.w)) ** 2).sum()
        l1_penalty = abs(self.w).sum()
        l2_penalty = TT.sqrt((self.w * self.w).sum())
        s_sparse_penalty = theano.dot(theano.dot(self.w.T, l_), self.w)

        loss = (
            e +
            self.lambda_1 * l1_penalty +
            self.lambda_2 * l2_penalty +
            self.alpha    * s_sparse_penalty
        ).sum()

        x_train, x_valid, y_train, y_valid = cv.train_test_split(_X, y)

        downhill.minimize(
            loss,
            XYLDataset(x_train, y_train, self.L, batch_size=self.batch_size),
            valid=XYLDataset(x_valid, y_valid, self.L, batch_size=x_valid.shape[0]),
            params=[self.w],
            inputs=[x_, y_, l_],
            algo="rmsprop",
            **self.downhill_args
        )
        w = self.w.get_value()
        self.coef_dist = [
            (abs(w) > x).sum() for x in [0.01, 0.001, 0.0001, 0.00001, 0.000001]
        ]

    def predict(self, X):
        if self.select_cols is not None:
            _X = X[:, self.select_cols]
        else:
            _X = X
        return np.dot(_X, self.w.get_value())
