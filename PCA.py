# Minimal PCA class required by ThinPlateSpline.py
# ECE 4370: Engineering for Surgery
# Fall 2026
# Replace with your own PCA class from an earlier module if you have one.

import numpy as np


class pca:
    def __init__(self, X):
        """
        X : [N x d] array of N points in d dimensions.

        Attributes
        ----------
        mean  : [d] mean of the points
        evals : [d] eigenvalues of the covariance matrix, sorted descending
        evecs : [d x d] eigenvectors as columns, matching evals
        """
        X = np.asarray(X, dtype=float)
        self.mean = np.mean(X, axis=0)
        Xc = X - self.mean[np.newaxis, :]
        cov = Xc.T @ Xc / max(X.shape[0], 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        self.evals = evals[order]
        self.evecs = evecs[:, order]
