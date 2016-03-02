import logging
import itertools as it

import numpy as np
import pandas as pd

import climate

import sklearn.cross_validation as cv
import sklearn.decomposition as decomp
import sklearn.linear_model as lm
import sklearn.pipeline as pipe
import sklearn.preprocessing as pre

import scipy.stats as stats

import data
import regressors as regr

np.random.seed(42)


def do_main():
    X, Y, V = data.get_XYV()
    wvs, has_wv = data.load_wvs(V)
    col_svd = decomp.TruncatedSVD(n_components=256).fit_transform(X.T)

    classifiers = [

        ("sparse", lambda: regr.SparseRegressor(lambda_1=4, lambda_2=.5)),

        # ("linear_regression", lambda: lm.LinearRegression()),
        # ("elastic_net", lambda: lm.ElasticNet()),
        # ("svd_64_linear_regression", lambda: pipe.Pipeline([
        #     ("decomp", decomp.TruncatedSVD(n_components=64)),
        #     ("linear_regression", lm.LinearRegression())
        # ])),

        # ("svd_128_linear_regression", lambda: pipe.Pipeline([
        #     ("decomp", decomp.TruncatedSVD(n_components=128)),
        #     ("linear_regression", lm.LinearRegression())
        # ])),

        # ("svd_256_linear_regression", lambda: pipe.Pipeline([
        #     ("decomp", decomp.TruncatedSVD(n_components=256)),
        #     ("linear_regression", lm.LinearRegression())
        # ])),
        # ("svd_512_linear_regression", lambda: pipe.Pipeline([
        #     ("decomp", decomp.TruncatedSVD(n_components=512)),
        #     ("linear_regression", lm.LinearRegression())
        # ])),

        (
            "sparse_doc_svd",
            lambda: regr.SSparseRegressor(
                col_svd,
                alpha=0.2,
                k_neighbors=5,
                lambda_1=4,
                lambda_2=.5
            )
        ),

        (
            "sparse_wv",
            lambda: regr.SSparseRegressor(
                wvs,
                alpha=0.2,
                k_neighbors=5,
                select_cols=has_wv,
                lambda_1=4,
                lambda_2=.5
            )
        )

    ]

    # these are the thresholds to test how large are the absolute values
    # of the coefficients
    gts = [0.01, 0.001, 0.0001, 1e-05, 1e-06]

    fname = "results_with_p.csv"
    try:
        df = pd.read_csv(fname)
    except IOError:
        df = pd.DataFrame(
            columns= [
                "task", "classifier", "score", "l1_norm", "p_val"
            ] + [
                "{:.6f}".format(x) for x in gts
            ]
        )

    for col in Y.columns:

        imputer = pre.Imputer()
        scaler = pre.StandardScaler()
        y = scaler.fit_transform(
            imputer.fit_transform(Y[col].as_matrix()[:, None])
        )

        for train_idx, test_idx in cv.ShuffleSplit(X.shape[0], 5, random_state=42):
            for classifier_name, classifier_f in classifiers:
                classifier = classifier_f()
                X_train = X[train_idx]
                X_test = X[test_idx]

                y_train = y[train_idx]
                y_test  = y[test_idx]

                classifier.fit(X_train, y_train)
                preds = classifier.predict(X_test)
                score, p_val = stats.pearsonr(y_test[:,0],  preds[:,0])
                w = classifier.w.get_value()
                l1_norm = np.linalg.norm(w, 1)
                size_dist = {"{:.6f}".format(x): (abs(w) >= x).sum() for x in gts}
                results = dict(
                    task=col,
                    classifier=classifier_name,
                    score=score,
                    l1_norm=l1_norm,
                    p_val=p_val
                )
                results.update(size_dist)
                df = df.append([results])
                logging.info("%s,%s,%s,%s,%s", col, classifier_name, score, l1_norm, p_val)
                df.to_csv(fname, index=False)

    df.to_csv(fname, index=False)


if __name__ == "__main__":
    climate.enable_default_logging()
    do_main()
