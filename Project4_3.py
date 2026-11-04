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

plt.show(block=True)