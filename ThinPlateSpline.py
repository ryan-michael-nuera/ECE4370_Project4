# Function to create thin plate spline deformation field
# ECE 5370: Engineering for Surgery
# Fall 2024
# Author: Prof.Jack Noble
# jack.noble @ vanderbilt.edu
import numpy as np
from PCA import * # replace this with path to your PCA class
def thinPlateSpline(source, target, dim, alpha=0, rbf='rlogsqrtr'):
    ndim = np.shape(source)[1]
    ps = pca(source)
    pt = pca(target)
    numlandmarks = np.shape(source)[0]
    if numlandmarks<ndim+1 or ps.evals[ndim-1]<1e-5 or pt.evals[ndim-1] <1e-5:
        print('Error: Need 4 or more non-planar landmarks to define TPS in 3D or 3 or more non-linear landmarks in 2D')
        return

    meshx = np.repeat(target[:, 0][np.newaxis,:], numlandmarks, axis=0)
    meshy = np.repeat(target[:, 1][np.newaxis,:], numlandmarks, axis=0)
    if ndim==3:
        meshz = np.repeat(target[:, 2][np.newaxis,:], numlandmarks, axis=0)

    r = (meshx - meshx.T)**2 + (meshy - meshy.T)**2
    if ndim==3:
        r+= (meshz - meshz.T)**2


    T = np.zeros((numlandmarks+ndim+1, numlandmarks+ndim+1))
    T[0: numlandmarks, 0] = 1
    T[0: numlandmarks, 1:(ndim+1)]= target
    if rbf=='rlogsqrtr':
        T[0: numlandmarks, (ndim+1): numlandmarks + (ndim+1)] = r * np.log(np.sqrt(r) + 1e-12) + alpha * np.eye(np.shape(r)[0])
    elif rbf == 'r':
        T[0: numlandmarks, (ndim + 1): numlandmarks + (ndim + 1)] = np.sqrt(r) + alpha * np.eye(
            np.shape(r)[0])
    T[numlandmarks, (ndim+1): numlandmarks + (ndim+1)] = 1
    T[numlandmarks + 1: numlandmarks + (ndim+1), (ndim+1): numlandmarks + (ndim+1)] = target.T

    a = np.linalg.solve(T, np.concatenate(((source[:, 0] - target[:, 0])[:,np.newaxis], np.zeros(((ndim+1), 1))), axis=0))
    b = np.linalg.solve(T, np.concatenate(((source[:, 1] - target[:, 1])[:,np.newaxis], np.zeros(((ndim+1), 1))), axis=0))
    if ndim==3:
        c = np.linalg.solve(T, np.concatenate(((source[:, 2] - target[:, 2])[:,np.newaxis], np.zeros((4, 1))), axis=0))

    x = np.linspace(0, dim[0] - 1, dim[0])
    y = np.linspace(0, dim[1] - 1, dim[1])
    if ndim==3:
        z = np.linspace(0, dim[2] - 1, dim[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    else:
        X, Y = np.meshgrid(x, y, indexing='ij')
    numpnts = np.size(X)
    r = (np.repeat(X.ravel()[:,np.newaxis], numlandmarks, axis=1) - np.repeat(target[:, 0][np.newaxis,:], numpnts, axis=0))**2 + \
        (np.repeat(Y.ravel()[:,np.newaxis], numlandmarks, axis=1) - np.repeat(target[:, 1][np.newaxis,:], numpnts, axis=0))**2
    if ndim==3:
        r += (np.repeat(Z.ravel()[:,np.newaxis], numlandmarks, axis=1) - np.repeat(target[:, 2][np.newaxis,:], numpnts, axis=0))**2
    pmat = np.ones((numpnts, (ndim+1) + numlandmarks))
    if ndim==3:
        if rbf == 'rlogsqrtr':
            pmat[:, 1: (ndim+1) + numlandmarks] = np.concatenate((X.ravel()[:,np.newaxis], Y.ravel()[:,np.newaxis],
                                                       Z.ravel()[:,np.newaxis], r * np.log(np.sqrt(r) + 1e-12)), axis=1)
        elif rbf == 'r':
            pmat[:, 1: (ndim + 1) + numlandmarks] = np.concatenate((X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis],
                                                                Z.ravel()[:, np.newaxis],
                                                                np.sqrt(r) ), axis=1)
    else:
        if rbf == 'rlogsqrtr':
            pmat[:, 1: (ndim + 1) + numlandmarks] = np.concatenate((X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis],
                                                                    r * np.log(np.sqrt(r) + 1e-12)), axis=1)
        elif rbf == 'r':
            pmat[:, 1: (ndim + 1) + numlandmarks] = np.concatenate((X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis],
                                                                np.sqrt(r) ), axis=1)
    Dx = np.reshape(pmat @ a, dim)
    Dy = np.reshape(pmat @ b, dim)
    if ndim==3:
        Dz = np.reshape(pmat @ c, dim)
        return Dx, Dy, Dz
    else:
        return Dx, Dy