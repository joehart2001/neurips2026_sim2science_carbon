"""Numerical routines extracted from /home/jh2536/amorphous-carbon/analysis_scripts/pore_analysis/identify_pores_PBC.py. Plotting removed."""
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm
def _merge_periodic_labels(labels, nx, ny, nz):
    """
    Merge labels that are connected across periodic boundaries.
    Only merge labels that represent the same connected component, not just if they touch on opposite sides of the box.
    """
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        if x == 0 or y == 0:  # Don't merge background
            return
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Check x-boundaries (left-right)
    for j in range(ny):
        for k in range(nz):
            left_label = labels[0, j, k]
            right_label = labels[-1, j, k]
            # Only merge if both are non-zero
            if left_label > 0 and right_label > 0:
                union(left_label, right_label)

    # Check y-boundaries (front-back)
    for i in range(nx):
        for k in range(nz):
            front_label = labels[i, 0, k]
            back_label = labels[i, -1, k]
            if front_label > 0 and back_label > 0:
                union(front_label, back_label)

    # Check z-boundaries (bottom-top)
    for i in range(nx):
        for j in range(ny):
            bottom_label = labels[i, j, 0]
            top_label = labels[i, j, -1]
            if bottom_label > 0 and top_label > 0:
                union(bottom_label, top_label)

    # Apply merging - vectorized for efficiency
    merged_labels = labels.copy()
    unique_labels = np.unique(labels[labels > 0])

    for label in unique_labels:
        root = find(label)
        if root != label:
            merged_labels[labels == label] = root

    return merged_labels

def _classify_pore_openness(mask, labels, pore_id, nx, ny, nz):
    """
    Classify if a pore is 'open' (spans from one side to opposite side) or 'closed'.

    A pore is spanning in a direction if:
    1. It touches both opposite faces in that direction
    2. Those face voxels are part of the same connected component (after periodic merging)

    Returns:
    --------
    pore_type : str
        'open' or 'closed'
    spanning_directions : list of str
        Which directions the pore spans (e.g., ['x', 'y'])
    """
    spanning_directions = []

    # Check x-direction (left to right)
    left_face = mask[0, :, :]
    right_face = mask[-1, :, :]

    if np.any(left_face) and np.any(right_face):
        # Since we've already merged periodic labels, if this pore touches both faces,
        # it spans in x-direction
        spanning_directions.append('x')

    # Check y-direction (front to back)
    front_face = mask[:, 0, :]
    back_face = mask[:, -1, :]

    if np.any(front_face) and np.any(back_face):
        spanning_directions.append('y')

    # Check z-direction (bottom to top)
    bottom_face = mask[:, :, 0]
    top_face = mask[:, :, -1]

    if np.any(bottom_face) and np.any(top_face):
        spanning_directions.append('z')

    pore_type = 'open' if len(spanning_directions) > 0 else 'closed'

    return pore_type, spanning_directions
