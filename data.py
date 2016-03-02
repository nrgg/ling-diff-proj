import scipy.io as io
import pandas as pd
import numpy as np
import os


def get_XYV():
    mat = io.loadmat(os.path.join("demo-data-distrib", "demo-data-distrib.mat"))
    columns = [x[0] for x in mat["header"][0]]
    Y = pd.DataFrame(mat["census"], columns=columns)
    V = pd.DataFrame([x[0] for x in mat["vocab"].T[0]], columns=["token"])
    return mat["counts"].toarray(), Y, V


def load_wvs(V, dim=50):
    tokens = dict(zip(V.token, V.index))
    wv_path = os.path.join("word-vectors","glove.twitter.27B.{dim}d.txt".format(dim=dim))
    wvs = np.zeros((len(tokens), dim))
    has_wv = [False] * len(tokens)
    for line in open(wv_path):
        tok = line.split(" ", 1)[0]
        if tok in tokens:
            wvs[tokens[tok], :] = map(float, line.split(" ")[1:])
            has_wv[tokens[tok]] = True
    return wvs, has_wv
