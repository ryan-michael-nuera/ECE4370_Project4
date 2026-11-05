# Script for Project 4 part 5
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Julia Ke; julia.ke@vanderbilt.edu
#           Sabra Winston; sabra.m.winston@vanderbilt.edu
#
# Manually select corresponding fiducials in ct (window 1) and t1 (window 2)
# with the Interactive_PointbasedImageRegistration class, rigidly register
# them, and compare the result to the ground truth transformation CT2T1.
#
# How to use the GUI:
#   1. Click "Add Fiducial" in each window, then left click a landmark in
#      any of the three slice views. Pick the SAME landmarks in the SAME
#      order in both windows (at least 4 well spread, non-coplanar points;
#      6 to 10 gives a better result).
#   2. Click "Rigid Register". Right click in a window to toggle between
#      that image and the other image projected onto it to check alignment.
#   3. Click "Exit". The selected fiducials are saved to
#      Project4_5_landmarks.json and loaded again next run, so the points
#      do not have to be re-picked.
#   4. Error metrics are printed and CT projected into T1 space with the
#      estimated transform is saved as Problem5_<view>_Comparison.png.

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from Interactive_PointBasedRegistration import interactive_pointBasedRegistration
from RigidPointRegister import rigidPointRegister
from Project4_2 import imageProjection
from Project4_4 import saveComparisonFigures

LANDMARK_FILE = "Project4_5_landmarks.json"


def loadLandmarks(fname):
    if not os.path.exists(fname):
        return None, None
    with open(fname, "r") as f:
        lm = json.load(f)
    print(f"Loaded {len(lm['ct'])} saved fiducial pairs from {fname}")
    return [list(p) for p in lm["ct"]], [list(p) for p in lm["t1"]]


def saveLandmarks(fname, l_ct, l_t1):
    with open(fname, "w") as f:
        json.dump({"ct": np.asarray(l_ct, dtype=float).tolist(),
                   "t1": np.asarray(l_t1, dtype=float).tolist()}, f, indent=2)
    print(f"Saved {len(l_ct)} fiducial pairs to {fname}")


def registerLandmarks(l_ct, l_t1, ctvoxsz, t1voxsz):
    """Rigid CT->T1 transform (mm) from voxel landmarks, as the class does."""
    p_ct = np.asarray(l_ct, dtype=float) * np.asarray(ctvoxsz)[np.newaxis, :]
    p_t1 = np.asarray(l_t1, dtype=float) * np.asarray(t1voxsz)[np.newaxis, :]
    p_ct2t1, T = rigidPointRegister(p_ct, p_t1)
    fre = np.sqrt(np.sum((p_ct2t1 - p_t1) ** 2, axis=1))
    return T, fre


def compareToGroundTruth(T_est, T_gt, ctdim, ctvoxsz):
    """Print error metrics between an estimated and ground truth CT->T1 transform."""
    # Residual transform: should be close to identity
    E = np.linalg.inv(T_gt) @ T_est

    R = E[:3, :3]
    cosang = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    rot_err_deg = np.degrees(np.arccos(cosang))
    trans_err = np.linalg.norm(E[:3, 3])

    # Displacement error over the whole CT volume (mm), on a coarse grid
    axes = [np.linspace(0, (n - 1) * v, 10) for n, v in zip(ctdim, ctvoxsz)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    pts = np.vstack((X.ravel(), Y.ravel(), Z.ravel(), np.ones(X.size)))
    disp = np.linalg.norm((T_est @ pts - T_gt @ pts)[:3], axis=0)

    np.set_printoptions(precision=4, suppress=True)
    print("\nEstimated CT -> T1 transform (T1to2):")
    print(T_est)
    print("\nGround truth CT2T1:")
    print(T_gt)
    print("\ninv(CT2T1) @ T1to2 (should be close to identity):")
    print(E)
    print(f"\nRotation error:            {rot_err_deg:.3f} deg")
    print(f"Translation component error: {trans_err:.3f} mm"
          f"  ({'PASS' if trans_err < 10 else 'FAIL'}: < 10 mm)")
    print(f"Error over CT volume:      mean {disp.mean():.3f} mm, "
          f"max {disp.max():.3f} mm")
    return E, rot_err_deg, trans_err


if __name__ == "__main__":
    with open("Project4.json", "r") as f:
        data = json.load(f)

    ct = np.array(data["ct"]["data"])
    ctvoxsz = np.array(data["ct"]["voxsz"], dtype=float)
    t1 = np.array(data["t1"]["data"])
    t1voxsz = np.array(data["t1"]["voxsz"], dtype=float)
    CT2T1 = np.array(data["CT2T1"], dtype=float)

    # Reload fiducials from a previous run if available
    l_ct, l_t1 = loadLandmarks(LANDMARK_FILE)

    # Window 1 = ct, window 2 = t1, so T1to2 maps CT mm -> T1 mm,
    # which is the same direction as CT2T1
    ipr = interactive_pointBasedRegistration(ct, ctvoxsz, t1, t1voxsz,
                                             landmarks1=l_ct, landmarks2=l_t1)

    l_ct = ipr.win1.landmarks
    l_t1 = ipr.win2.landmarks
    plt.close("all")

    if len(l_ct) != len(l_t1):
        raise SystemExit(f"Unequal number of fiducials: ct has {len(l_ct)}, "
                         f"t1 has {len(l_t1)}. Rerun and fix the point lists.")
    if len(l_ct) < 4:
        raise SystemExit("Select at least 4 fiducial pairs.")

    saveLandmarks(LANDMARK_FILE, l_ct, l_t1)

    # Use the class result if "Rigid Register" was clicked, otherwise
    # compute the same registration from the selected fiducials
    T_est, fre = registerLandmarks(l_ct, l_t1, ctvoxsz, t1voxsz)
    if np.size(ipr.T1to2) > 0:
        T_est = np.asarray(ipr.T1to2)

    print(f"\nFiducial registration error: mean {fre.mean():.3f} mm, "
          f"max {fre.max():.3f} mm over {len(fre)} fiducials")
    compareToGroundTruth(T_est, CT2T1, ct.shape, ctvoxsz)

    # Project CT onto the T1 grid with the estimated transform
    # (imageProjection needs the target -> source map, i.e. T1 -> CT)
    ct_projected = imageProjection(
        source=ct,
        targetdim=np.array(t1.shape),
        sourcevoxsz=ctvoxsz,
        targetvoxsz=t1voxsz,
        T=np.linalg.inv(T_est)
    )
    saveComparisonFigures(t1.astype(float), ct_projected, t1voxsz,
                          "Original T1 MRI", "CT (point-based registration)",
                          "Problem5")
    plt.show(block=True)
