# Script for Project 4 part 1
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Julia Ke; julia.ke@vanderbilt.edu
#           Sabra Winston; sabra.m.winston@vanderbilt.edu
#
# Apply a sequence of three rigid coordinate transformations to the
# landmark set 'Proj4landmarks'. Each transformation rotates about x, then
# y, then z (R = Rz @ Ry @ Rx) by the angles in theta (degrees), then
# translates by t (mm):  x' = R x + t.

import json
import numpy as np


def rotationMatrix(theta_deg):
    """R = Rz @ Ry @ Rx for angles [ax, ay, az] in degrees."""
    ax, ay, az = np.radians(theta_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def rigidTransform(theta_deg, t_mm):
    """4x4 homogeneous transform: rotate by theta, then translate by t."""
    T = np.eye(4)
    T[:3, :3] = rotationMatrix(theta_deg)
    T[:3, 3] = t_mm
    return T


def applyTransform(T, pnts):
    """Apply 4x4 transform T to an N x 3 array of points."""
    pnts = np.asarray(pnts, dtype=float)
    homog = np.hstack((pnts, np.ones((pnts.shape[0], 1))))
    return (T @ homog.T).T[:, :3]


if __name__ == "__main__":
    with open("Project4.json", "r") as f:
        data = json.load(f)
    landmarks = np.array(data["Proj4landmarks"], dtype=float)

    # Transformation parameters: rotation angles (degrees), translation (mm)
    params = [
        ([10, -15, 5], [2, -1, 4]),      # Transformation 1
        ([5, 10, 2], [-1, 4, 0]),        # Transformation 2
        ([5, 5, -7], [-0.5, -3, -4]),    # Transformation 3
    ]

    # Compose in order: T3 @ T2 @ T1 (Transformation 1 is applied first)
    T_total = np.eye(4)
    for i, (theta, t) in enumerate(params, start=1):
        Ti = rigidTransform(theta, t)
        T_total = Ti @ T_total
        print(f"Transformation {i} (theta = {theta} deg, t = {t} mm):")
        print(np.round(Ti, 6))

    print("\nComposite transformation T3 @ T2 @ T1:")
    print(np.round(T_total, 6))

    result = applyTransform(T_total, landmarks)

    np.set_printoptions(precision=8, suppress=True)
    print("\nOriginal landmarks (mm):")
    print(landmarks)
    print("\nTransformed landmarks (mm):")
    print(result)

    hint = np.array([0.63815515, -0.82001863, 0.38204109])
    err = np.linalg.norm(result[0] - hint)
    print(f"\nFirst point vs hint {hint}: difference {err:.2e} mm "
          f"({'PASS' if err < 1e-6 else 'CHECK'})")
