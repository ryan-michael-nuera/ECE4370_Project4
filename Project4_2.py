# Script for Project 4 part 2
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Julia Ke; julia.ke@vanderbilt.edu
#           Sabra Winston; sabra.m.winston@vanderbilt.edu



import numpy as np
from scipy.ndimage import map_coordinates

def imageProjection(source, targetdim, sourcevoxsz=[],
                    targetvoxsz=[], T=[], D=[]):
    """
    Project a 2D or 3D source image onto a target grid.

    Parameters
    ----------
    source : numpy.ndarray
        Source image.

    targetdim : array-like
        Dimensions of the target image.

    sourcevoxsz : array-like
        Source voxel sizes in mm.

    targetvoxsz : array-like
        Target voxel sizes in mm.

    T : numpy.ndarray
        4x4 transformation matrix mapping target
        coordinates to source coordinates in mm.

    D : numpy.ndarray
        Deformation field with shape (3, r, c, d).
        Displacements are measured in voxels.

    Returns
    -------
    target : numpy.ndarray
        Projected image with dimensions targetdim.
    """

    source = np.asarray(source)
    targetdim = tuple(np.asarray(targetdim, dtype=int))

    if source.ndim not in (2, 3):
        raise ValueError("Source must be a 2D or 3D image.")

    if len(targetdim) != source.ndim:
        raise ValueError("Target dimensions must match source dimensionality.")

    # Create coordinates for every voxel in target
    grid = np.meshgrid(
        *[np.arange(n) for n in targetdim],
        indexing="ij"
    )

    coords = np.array([g.ravel() for g in grid], dtype=float)

    has_T = np.asarray(T).size > 0
    has_D = np.asarray(D).size > 0

    if has_T and has_D:
        raise ValueError("Specify either T or D, not both.")

    if has_T:
        # Transformation matrix uses millimeters
        T = np.asarray(T, dtype=float)
        sourcevoxsz = np.asarray(sourcevoxsz, dtype=float)
        targetvoxsz = np.asarray(targetvoxsz, dtype=float)

        ndim = source.ndim

        if T.shape != (ndim + 1, ndim + 1) and T.shape != (4, 4):
            raise ValueError("Invalid transformation matrix shape.")

        if sourcevoxsz.size < ndim or targetvoxsz.size < ndim:
            raise ValueError("Voxel sizes are required when using T.")

        # Convert target voxel coordinates to mm
        target_mm = coords * targetvoxsz[:ndim, None]

        # Add homogeneous coordinate
        homogeneous = np.vstack(
            (target_mm, np.ones(target_mm.shape[1]))
        )

        # Apply transformation in mm
        if T.shape == (4, 4) and ndim == 2:
            homogeneous = np.vstack(
                (target_mm, np.zeros(target_mm.shape[1]),
                 np.ones(target_mm.shape[1]))
            )

        source_mm = T @ homogeneous

        # Convert source mm coordinates to voxel indices
        source_coords = (
            source_mm[:ndim] / sourcevoxsz[:ndim, None]
        )

    elif has_D:
        D = np.asarray(D, dtype=float)

        if D.shape != (3,) + targetdim:
            raise ValueError(
                "Deformation field must have shape (3, r, c, d)."
            )

        if source.ndim != 3:
            raise ValueError("Deformation field requires a 3D image.")

        # Add deformation displacement to target coordinates
        source_coords = coords + D.reshape(3, -1)

    else:
        # No transformation: use identity voxel mapping
        source_coords = coords

    # Interpolate source image at transformed positions
    values = map_coordinates(
        source.astype(float),
        source_coords,
        order=1,
        mode="constant",
        cval=0.0
    )

    return values.reshape(targetdim)
