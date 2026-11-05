# Script for Project 4 part 1
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Julia Ke; julia.ke@vanderbilt.edu
#           Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Sabra Winston; sabra.m.winston@vanderbilt.edu
# Coordinate Transformations

import json
import numpy as np


def rotation_matrix(theta):
    """
    Construct a 3D rotation matrix from rotations about
    x, y, and z.

    theta = [theta_x, theta_y, theta_z] in degrees
    """

    theta = np.deg2rad(np.asarray(theta, dtype=float))

    tx = theta[0]
    ty = theta[1]
    tz = theta[2]

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(tx), -np.sin(tx)],
        [0, np.sin(tx),  np.cos(tx)]
    ])

    Ry = np.array([
        [ np.cos(ty), 0, np.sin(ty)],
        [0,           1, 0],
        [-np.sin(ty), 0, np.cos(ty)]
    ])

    Rz = np.array([
        [np.cos(tz), -np.sin(tz), 0],
        [np.sin(tz),  np.cos(tz), 0],
        [0,           0,          1]
    ])

    return Rz @ Ry @ Rx


def make_transform(theta, translation):
    """
    Construct a 4x4 homogeneous transformation matrix.

    theta       : [rx, ry, rz] degrees
    translation : [tx, ty, tz] mm
    """

    T = np.eye(4)

    T[:3, :3] = rotation_matrix(theta)
    T[:3, 3] = np.asarray(translation, dtype=float)

    return T


def transform_points(points, T):
    """
    Transform an Nx3 array of points using a 4x4
    homogeneous transformation matrix.
    """

    points = np.asarray(points, dtype=float)

    # Convert Nx3 -> Nx4 homogeneous coordinates
    points_h = np.column_stack(
        (points, np.ones(points.shape[0]))
    )

    # Apply transformation
    transformed_h = (T @ points_h.T).T

    return transformed_h[:, :3]


# ============================================================
# Load Project4.json
# ============================================================

with open("Project4.json", "r") as f:
    data = json.load(f)

landmarks = np.asarray(
    data["Proj4landmarks"],
    dtype=float
)


# ============================================================
# Transformation parameters from the assignment
# ============================================================

theta1 = np.array([
    10,
    -15,
    5
], dtype=float)

t1 = np.array([
    2,
    -1,
    4
], dtype=float)


theta2 = np.array([
    5,
    10,
    2
], dtype=float)

t2 = np.array([
    -1,
    4,
    0
], dtype=float)


theta3 = np.array([
    2,
    5,
    -7
], dtype=float)

t3 = np.array([
    -0.5,
    -3,
    -4
], dtype=float)


# ============================================================
# Construct transformation matrices
# ============================================================

T1 = make_transform(theta1, t1)
T2 = make_transform(theta2, t2)
T3 = make_transform(theta3, t3)


# ============================================================
# Perform transformations SEQUENTIALLY
#
# p1 = T1 p
# p2 = T2 p1
# p3 = T3 p2
# ============================================================

points_after_T1 = transform_points(
    landmarks,
    T1
)

points_after_T2 = transform_points(
    points_after_T1,
    T2
)

final_points = transform_points(
    points_after_T2,
    T3
)


# Equivalent combined transformation:
#
# Ttotal = T3 @ T2 @ T1
#
# final_points = transform_points(landmarks, Ttotal)

Ttotal = T3 @ T2 @ T1


# ============================================================
# Print results
# ============================================================

np.set_printoptions(
    precision=8,
    suppress=True
)

print("\nOriginal landmarks:")
print(landmarks)

print("\nTransformation 1:")
print(T1)

print("\nTransformation 2:")
print(T2)

print("\nTransformation 3:")
print(T3)

print("\nCombined transformation T3 @ T2 @ T1:")
print(Ttotal)

print("\nFinal transformed landmarks:")
print(final_points)


# ============================================================
# Assignment hint check
# ============================================================

expected = np.array([
    0.63815515,
    -0.82001863,
    0.38204109
])

print("\nFirst transformed coordinate:")
print(final_points[0])

print("\nExpected coordinate:")
print(expected)

print("\nDifference:")
print(final_points[0] - expected)

print(
    "\nMatches assignment hint:",
    np.allclose(
        final_points[0],
        expected,
        atol=1e-6
    )
)