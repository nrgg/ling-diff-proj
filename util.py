import scipy.sparse as sparse
import sklearn.metrics.pairwise as pairwise


def symmetrify_similarity(g, vecs):
    rows, cols = g.nonzero()
    gdense = g.toarray()
    for row, col in zip(rows, cols):
        sim = pairwise.cosine_similarity([vecs[row], vecs[col]])[0, 1]
        gdense[col, row] = gdense[row, col] = sim
    return sparse.csr_matrix(gdense)
