import numpy as np
import json

from Project4_2 import imageProjection
from volumeViewer import volumeViewer
import matplotlib.pyplot as plt

# Load Project 4 dataset
with open("Project4.json", "r") as f:
    data = json.load(f)

# CT image
ct = np.array(data["ct"]["data"])
ctvoxsz = np.array(data["ct"]["voxsz"])

# T1 MRI image
t1 = np.array(data["t1"]["data"])
t1voxsz = np.array(data["t1"]["voxsz"])

# Transformation matrix
CT2T1 = np.array(data["CT2T1"])

# Convert CT-to-T1 into T1-to-CT
T1toCT = np.linalg.inv(CT2T1)

# Project CT onto the T1 voxel grid
ct_projected = imageProjection(
    source=ct,
    targetdim=np.array(t1.shape),
    sourcevoxsz=ctvoxsz,
    targetvoxsz=t1voxsz,
    T=T1toCT
)

print("Original CT shape:", ct.shape)
print("T1 shape:", t1.shape)
print("Projected CT shape:", ct_projected.shape)

# Display original T1
viewer_t1 = volumeViewer("Original T1 MRI")
viewer_t1.setImage(
    t1,
    t1voxsz,
    contrast=1000,
    level=0
)

# Display projected CT
viewer_ct = volumeViewer("CT Projected into T1 Space")
viewer_ct.setImage(
    ct_projected,
    t1voxsz,
    contrast=1000,
    level=0
)

# Update all three anatomical views
for i in range(3):
    viewer_t1.update(direction=i)
    viewer_ct.update(direction=i)

# Compare CT and T1 with physical voxel spacing
import numpy as np
import matplotlib.pyplot as plt

slices = [
    t1.shape[2] // 2,
    t1.shape[1] // 2,
    t1.shape[0] // 2
]

view_names = ["Axial", "Coronal", "Sagittal"]

# Physical aspect ratios for each view
aspect_ratios = [
    t1voxsz[0] / t1voxsz[1],  # Axial
    t1voxsz[0] / t1voxsz[2],  # Coronal
    t1voxsz[1] / t1voxsz[2]   # Sagittal
]

for i, (view, idx) in enumerate(zip(view_names, slices)):

    if i == 0:
        t1_slice = t1[:, :, idx]
        ct_slice = ct_projected[:, :, idx]

    elif i == 1:
        t1_slice = t1[:, idx, :]
        ct_slice = ct_projected[:, idx, :]

    else:
        t1_slice = t1[idx, :, :]
        ct_slice = ct_projected[idx, :, :]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original T1
    axes[0].imshow(
        t1_slice,
        cmap="gray",
        aspect=aspect_ratios[i]
    )
    axes[0].set_title(f"{view} - Original T1 MRI")
    axes[0].axis("off")

    # CT/T1 overlay
    axes[1].imshow(
        t1_slice,
        cmap="gray",
        aspect=aspect_ratios[i]
    )

    ct_masked = np.ma.masked_where(
        ct_slice == 0,
        ct_slice
    )

    axes[1].imshow(
        ct_masked,
        cmap="hot",
        alpha=0.35,
        aspect=aspect_ratios[i]
    )

    axes[1].set_title(f"{view} - CT/T1 Overlay")
    axes[1].axis("off")

    fig.suptitle(
        f"{view} View - CT to T1 Registration",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        f"Problem3_{view}_Comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show(block=True)

