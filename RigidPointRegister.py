# Function to perform rigid or rigid + scaling point registration
# ECE 4370/5370: Engineering for Surgery
# Fall 2024
# Author: Prof. Jack Noble; jack.noble@vanderbilt.edu
# source and target are [N x 2] or [N x 3] matrices
# reflection = True permits non-proper rotations
# scaling = True adds isotropic scaling
# returns source_to_target (source points projected to target coordinate frame),
#         T_stot (The homogeneous coordinate transformation from the source to the target coordinate frame)
import numpy as np
def rigidPointRegister(source, target, reflection=False, scaling=False):

    ndim = np.shape(source)[1]
    npts = np.shape(source)[0]
    source_mn = np.mean(source, axis=0)
    target_mn = np.mean(target, axis=0)

    covmat = (target-target_mn[np.newaxis,:]).T @ (source-source_mn[np.newaxis,:]) / npts
    U,S,Vt = np.linalg.svd(covmat)
    if reflection==False:
        D = np.eye(np.shape(U)[0])
        D[-1,-1]= np.linalg.det(Vt.T @ U)
        R= U @ D @ Vt
    else:
        R = U @ Vt

    if scaling:
        tt = target-target_mn[np.newaxis,:]
        st = source-source_mn[np.newaxis,:]
        s = np.sum((R @ st.T)*tt.T)/np.sum(st*st)
    else:
        s=1

    T_stot = np.zeros((ndim+1, ndim+1))
    T_stot[0:ndim,0:ndim] = s*R
    T_stot[0:ndim,ndim] = target_mn.T-s*(R @ source_mn.T)
    T_stot[ndim,ndim] = 1
    source_to_target = ((T_stot @ np.concatenate((np.array(source).T,np.ones((1,npts))),axis=0))[0:ndim,:]).T
    return source_to_target, T_stot