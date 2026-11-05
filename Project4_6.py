# Script for Project 4 part 6
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Julia Ke; julia.ke@vanderbilt.edu
#           Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Sabra Winston; sabra.m.winston@vanderbilt.edu
# Thin Plate Spline registration: ctwarp -> ct

import json
import numpy as np
import matplotlib.pyplot as plt

from Interactive_PointBasedRegistration import interactive_pointBasedRegistration


def load_deformation_field(D, target_shape):
    """
    Make sure D has shape (3, r, c, d).
    """
    D = np.asarray(D, dtype=float)
    target_shape = tuple(target_shape)

    if D.shape == (3,) + target_shape:
        return D

    if D.shape == target_shape + (3,):
        return np.moveaxis(D, -1, 0)

    raise ValueError(
        f"Unexpected D shape: {D.shape}\n"
        f"Expected {(3,) + target_shape} "
        f"or {target_shape + (3,)}"
    )


# ============================================================
# Load Project4 data
# ============================================================

with open("Project4.json", "r") as f:
    data = json.load(f)

ctwarp = np.array(data["ctwarp"]["data"])
ct = np.array(data["ct"]["data"])

# Handle either naming convention for CT voxel size
if "voxsz" in data["ct"]:
    ctvoxsz = np.array(data["ct"]["voxsz"], dtype=float)
elif "sz" in data["ct"]:
    ctvoxsz = np.array(data["ct"]["sz"], dtype=float)
else:
    raise KeyError("Could not find CT voxel size ('voxsz' or 'sz').")

# Ground-truth deformation field
D_gt = load_deformation_field(data["D"], ct.shape)

print("ctwarp shape:", ctwarp.shape)
print("ct shape:    ", ct.shape)
print("D shape:     ", D_gt.shape)
print("CT voxel size:", ctvoxsz)


# ============================================================
# Interactive TPS registration
#
# Window 1 = ctwarp
# Window 2 = ct
#
# Instructions:
#   1. Click "Add Fiducial" in BOTH windows.
#   2. Select the same anatomical point in both images.
#   3. Repeat for several corresponding points.
#   4. Use points spread throughout the image volume.
#   5. Click "TPS Register".
#   6. Check registration by right-clicking to toggle images.
#   7. Click "Exit" when finished.
# ============================================================

ipr = interactive_pointBasedRegistration(
    ctwarp,
    ctvoxsz,
    ct,
    ctvoxsz
)

plt.close("all")


# ============================================================
# Make sure TPS Register was clicked
# ============================================================

if np.size(ipr.Db) == 0:
    raise RuntimeError(
        "\nNo backward deformation field was generated.\n"
        "Run the program again and make sure you click "
        "'TPS Register' BEFORE clicking 'Exit'."
    )


# ============================================================
# Get estimated backward deformation field
# ============================================================

Db = np.asarray(ipr.Db, dtype=float)

print("\nEstimated Db shape:", Db.shape)
print("Ground truth D shape:", D_gt.shape)

if Db.shape != D_gt.shape:
    raise ValueError(
        f"Shape mismatch:\n"
        f"Db = {Db.shape}\n"
        f"D  = {D_gt.shape}"
    )


# ============================================================
# Compare estimated deformation with ground truth
# ============================================================

difference = Db - D_gt
abs_difference = np.abs(difference)

mean_abs_difference = np.mean(abs_difference)

x_error = np.mean(abs_difference[0])
y_error = np.mean(abs_difference[1])
z_error = np.mean(abs_difference[2])

max_error = np.max(abs_difference)


# ============================================================
# Print results
# ============================================================

print("\n==========================================")
print("PROJECT 4 - PART 6 RESULTS")
print("==========================================")

print(
    f"Mean absolute difference = "
    f"{mean_abs_difference:.4f} voxels"
)

print("\nMean absolute error by component:")
print(f"X = {x_error:.4f} voxels")
print(f"Y = {y_error:.4f} voxels")
print(f"Z = {z_error:.4f} voxels")

print(f"\nMaximum absolute difference = {max_error:.4f} voxels")

print("------------------------------------------")

if mean_abs_difference <= 10:
    print("PASS")
    print("Mean absolute difference <= 10 voxels.")
else:
    print("FAIL")
    print("Mean absolute difference > 10 voxels.")
    print(
        "Try selecting more accurate corresponding points "
        "and distribute them throughout the image volume."
    )

print("==========================================")


# ============================================================
# Optional: save Db so you do not lose the result
# ============================================================

np.save("Project4_Part6_Db.npy", Db)

print("\nSaved estimated backward deformation field as:")
print("Project4_Part6_Db.npy")