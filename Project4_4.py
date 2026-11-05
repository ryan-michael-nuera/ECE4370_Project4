# Script for Project 4 part 4
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Julia Ke; julia.ke@vanderbilt.edu
#           Sabra Winston; sabra.m.winston@vanderbilt.edu
#
# Project the ctwarp image into ct space using the provided deformation
# field D, then compare the result to ct with the volumeViewer class.

import json
import numpy as np
import matplotlib.pyplot as plt

from Project4_2 import imageProjection
from volumeViewer import volumeViewer


def loadDeformationField(D, targetdim):
    """Return D as a 3 x r x c x d array (voxel units)."""
    D = np.asarray(D, dtype=float)
    targetdim = tuple(int(n) for n in targetdim)
    if D.shape == (3,) + targetdim:
        return D
    if D.shape == targetdim + (3,):
        # stored as r x c x d x 3 -> move components to the first axis
        return np.moveaxis(D, -1, 0)
    raise ValueError(f"Unexpected deformation field shape {D.shape}; "
                     f"expected (3,)+{targetdim}")


def saveComparisonFigures(ref, moving, voxsz, ref_name, moving_name,
                          prefix, vmin=None, vmax=None):
    """
    Save one figure per view (axial, coronal, sagittal) with three panels:
    reference image, projected image, and an overlay of the two.
    Voxels outside the projected field of view (value 0) are not overlaid.
    """
    slices = [ref.shape[2] // 2, ref.shape[1] // 2, ref.shape[0] // 2]
    view_names = ["Axial", "Coronal", "Sagittal"]

    # Physical aspect ratios for each view
    aspect_ratios = [
        voxsz[0] / voxsz[1],  # Axial
        voxsz[0] / voxsz[2],  # Coronal
        voxsz[1] / voxsz[2]   # Sagittal
    ]

    for i, (view, idx) in enumerate(zip(view_names, slices)):
        if i == 0:
            ref_slice = ref[:, :, idx]
            mov_slice = moving[:, :, idx]
        elif i == 1:
            ref_slice = ref[:, idx, :]
            mov_slice = moving[:, idx, :]
        else:
            ref_slice = ref[idx, :, :]
            mov_slice = moving[idx, :, :]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(ref_slice, cmap="gray", vmin=vmin, vmax=vmax,
                       aspect=aspect_ratios[i])
        axes[0].set_title(f"{view} - {ref_name}")
        axes[0].axis("off")

        axes[1].imshow(mov_slice, cmap="gray", vmin=vmin, vmax=vmax,
                       aspect=aspect_ratios[i])
        axes[1].set_title(f"{view} - {moving_name}")
        axes[1].axis("off")

        axes[2].imshow(ref_slice, cmap="gray", vmin=vmin, vmax=vmax,
                       aspect=aspect_ratios[i])
        mov_masked = np.ma.masked_where(mov_slice == 0, mov_slice)
        axes[2].imshow(mov_masked, cmap="hot", alpha=0.35, vmin=vmin,
                       vmax=vmax, aspect=aspect_ratios[i])
        axes[2].set_title(f"{view} - Overlay")
        axes[2].axis("off")

        fig.suptitle(f"{view} View - {moving_name} vs {ref_name}",
                     fontsize=14, y=1.03)
        plt.tight_layout()
        plt.savefig(f"{prefix}_{view}_Comparison.png", dpi=300,
                    bbox_inches="tight")


if __name__ == "__main__":
    with open("Project4.json", "r") as f:
        data = json.load(f)

    # CT image (target space)
    ct = np.array(data["ct"]["data"])
    ctvoxsz = np.array(data["ct"]["voxsz"])

    # Warped CT image (source image to be projected)
    ctwarp = np.array(data["ctwarp"]["data"])

    # Deformation field: maps ct voxel coordinates to ctwarp voxel coordinates
    D = loadDeformationField(data["D"], ct.shape)

    # Project ctwarp onto the ct voxel grid. Only D is used, so the
    # result lives in ct space and needs no voxel sizes.
    ctwarp_projected = imageProjection(
        source=ctwarp,
        targetdim=np.array(ct.shape),
        D=D
    )

    print("ctwarp shape:", ctwarp.shape)
    print("ct shape:", ct.shape)
    print("Projected ctwarp shape:", ctwarp_projected.shape)

    # Quantitative check: agreement with ct before and after projection
    ctf = ct.astype(float)
    before = np.mean(np.abs(ctwarp.astype(float) - ctf))
    after = np.mean(np.abs(ctwarp_projected - ctf))
    corr_before = np.corrcoef(ctwarp.ravel(), ctf.ravel())[0, 1]
    corr_after = np.corrcoef(ctwarp_projected.ravel(), ctf.ravel())[0, 1]
    print(f"Mean |ctwarp - ct|            before: {before:.2f}   "
          f"correlation: {corr_before:.3f}")
    print(f"Mean |projected ctwarp - ct|  after:  {after:.2f}   "
          f"correlation: {corr_after:.3f}")

    # Display original CT
    viewer_ct = volumeViewer("Original CT")
    viewer_ct.setImage(ct, ctvoxsz, contrast=1000, level=0)

    # Display warped CT projected into CT space
    viewer_proj = volumeViewer("ctwarp Projected into CT Space")
    viewer_proj.setImage(ctwarp_projected, ctvoxsz, contrast=1000, level=0)

    # Update all three anatomical views
    for i in range(3):
        viewer_ct.update(direction=i)
        viewer_proj.update(direction=i)

    # Save CT vs projected ctwarp comparison figures
    # (window of -500 to 500 HU, matching contrast=1000, level=0)
    saveComparisonFigures(ctf, ctwarp_projected, ctvoxsz,
                          "Original CT", "Projected ctwarp",
                          "Problem4", vmin=-500, vmax=500)

    plt.show(block=True)
