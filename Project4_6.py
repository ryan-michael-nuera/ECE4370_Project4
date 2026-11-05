# Script for Project 4 part 6
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Julia Ke; julia.ke@vanderbilt.edu
#           Sabra Winston; sabra.m.winston@vanderbilt.edu
#
# Manually select corresponding fiducials in ctwarp (window 1) and ct
# (window 2) with the Interactive_PointbasedImageRegistration class, run
# thin plate spline (TPS) registration, and compare the backward deformation
# field Db (ct grid -> ctwarp voxel coordinates) with the ground truth D.
#
# How to use the GUI:
#   1. Click "Add Fiducial" in each window, then left click a landmark in the
#      axial view. Pick the SAME landmarks in the SAME order in both windows.
#      Use 10 to 15 points spread over the whole head (low, middle and high
#      slices; front, back, left and right). TPS memory use grows with the
#      number of points (about 3.5 GB at 12 points for this image size).
#   2. Click "TPS Register". Right click in a window to toggle between that
#      image and the other image warped onto it.
#   3. Click "Exit". The fiducials are saved to Project4_6_landmarks.json and
#      loaded again next run.
#   4. The comparison of Db with D is printed, and figures are saved as
#      Problem6_<view>_Comparison.png and Problem6_Db_vs_D.png.

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from Interactive_PointBasedRegistration import interactive_pointBasedRegistration
from ThinPlateSpline import thinPlateSpline
from Project4_2 import imageProjection
from Project4_4 import loadDeformationField, saveComparisonFigures

LANDMARK_FILE = "Project4_6_landmarks.json"


def loadLandmarks(fname):
    if not os.path.exists(fname):
        return None, None
    with open(fname, "r") as f:
        lm = json.load(f)
    print(f"Loaded {len(lm['ctwarp'])} saved fiducial pairs from {fname}")
    return [list(p) for p in lm["ctwarp"]], [list(p) for p in lm["ct"]]


def saveLandmarks(fname, l_cw, l_ct):
    with open(fname, "w") as f:
        json.dump({"ctwarp": np.asarray(l_cw, dtype=float).tolist(),
                   "ct": np.asarray(l_ct, dtype=float).tolist()}, f, indent=2)
    print(f"Saved {len(l_cw)} fiducial pairs to {fname}")


def compareDeformationFields(Db, D, mask=None):
    """Print and return error metrics between estimated Db and ground truth D."""
    diff = np.abs(Db - D)
    mad = diff.mean()
    mad_xyz = diff.reshape(3, -1).mean(axis=1)
    eucl = np.sqrt(((Db - D) ** 2).sum(axis=0))

    print("\nDb vs ground truth D (voxel units):")
    print(f"Mean absolute difference (all components): {mad:.3f} voxels"
          f"  ({'PASS' if mad <= 10 else 'FAIL'}: <= 10 voxels)")
    print(f"Mean absolute difference x, y, z:          "
          f"{mad_xyz[0]:.3f}, {mad_xyz[1]:.3f}, {mad_xyz[2]:.3f} voxels")
    print(f"Mean / max vector error magnitude:         "
          f"{eucl.mean():.3f} / {eucl.max():.3f} voxels")
    if mask is not None:
        print(f"Mean absolute difference inside head:      "
              f"{diff[:, mask].mean():.3f} voxels")
    # Reference: a zero field (no registration) for scale
    print(f"Mean |D| (error of a zero field, reference): "
          f"{np.abs(D).mean():.3f} voxels")
    return mad


def saveFieldFigure(Db, D, fname, z=None):
    """Axial slice of each component of D and Db, plus their difference."""
    if z is None:
        z = D.shape[3] // 2
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    names = ["x", "y", "z"]
    for c in range(3):
        lim = np.abs(D[c, :, :, z]).max()
        for col, (img, title, cmap, vl) in enumerate([
                (D[c, :, :, z], f"D{names[c]} (ground truth)", "coolwarm", lim),
                (Db[c, :, :, z], f"Db{names[c]} (TPS)", "coolwarm", lim),
                (np.abs(Db[c, :, :, z] - D[c, :, :, z]), f"|Db{names[c]} - D{names[c]}|",
                 "magma", lim)]):
            vmin = 0 if col == 2 else -vl
            im = axes[c, col].imshow(img.T, cmap=cmap, vmin=vmin, vmax=vl)
            axes[c, col].set_title(title)
            axes[c, col].axis("off")
            fig.colorbar(im, ax=axes[c, col], fraction=0.046, label="voxels")
    fig.suptitle(f"Deformation field comparison, axial slice z = {z}", fontsize=14)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    with open("Project4.json", "r") as f:
        data = json.load(f)

    ct = np.array(data["ct"]["data"])
    ctvoxsz = np.array(data["ct"]["voxsz"], dtype=float)
    ctwarp = np.array(data["ctwarp"]["data"])
    D = loadDeformationField(data["D"], ct.shape)

    # Reload fiducials from a previous run if available
    l_cw, l_ct = loadLandmarks(LANDMARK_FILE)

    # Window 1 = ctwarp, window 2 = ct, so the class member Db is defined on
    # the ct grid and points to ctwarp voxel coordinates, the same as D
    ipr = interactive_pointBasedRegistration(ctwarp, ctvoxsz, ct, ctvoxsz,
                                             landmarks1=l_cw, landmarks2=l_ct)

    l_cw = ipr.win1.landmarks
    l_ct = ipr.win2.landmarks
    plt.close("all")

    if len(l_cw) != len(l_ct):
        raise SystemExit(f"Unequal number of fiducials: ctwarp has {len(l_cw)}, "
                         f"ct has {len(l_ct)}. Rerun and fix the point lists.")
    if len(l_cw) < 4:
        raise SystemExit("Select at least 4 non-planar fiducial pairs.")

    saveLandmarks(LANDMARK_FILE, l_cw, l_ct)

    # Use the class result if "TPS Register" was clicked, otherwise compute
    # the same backward field from the selected fiducials
    if np.size(ipr.Db) > 0:
        Db = np.asarray(ipr.Db)
    else:
        Dx, Dy, Dz = thinPlateSpline(np.array(l_cw, dtype=float),
                                     np.array(l_ct, dtype=float), ct.shape)
        Db = np.stack((Dx, Dy, Dz))
    del ipr

    head = ct > -500
    compareDeformationFields(Db, D, mask=head)

    # Warp ctwarp into ct space with Db and compare with the original ct
    ctwarp_tps = imageProjection(ctwarp, np.array(ct.shape), D=Db)
    ctf = ct.astype(float)
    corr_before = np.corrcoef(ctwarp.ravel(), ctf.ravel())[0, 1]
    corr_after = np.corrcoef(ctwarp_tps.ravel(), ctf.ravel())[0, 1]
    print(f"\nCorrelation with ct: ctwarp {corr_before:.3f}, "
          f"ctwarp warped with Db {corr_after:.3f}")

    saveComparisonFigures(ctf, ctwarp_tps, ctvoxsz, "Original CT",
                          "ctwarp (TPS registration)", "Problem6",
                          vmin=-500, vmax=500)
    saveFieldFigure(Db, D, "Problem6_Db_vs_D.png",
                    z=int(np.median(np.array(l_ct)[:, 2])))
    plt.show(block=True)
