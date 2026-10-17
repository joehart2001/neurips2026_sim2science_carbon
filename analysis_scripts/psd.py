"""Sphere-volume-weighted pore-size distribution for panel (d), using PoreTorch."""

import numpy as np
import torch
from poretorch import field_from_atoms, label_pores, pore_size_distribution
from poretorch.psd import bin_distribution


def calculate_psd(atoms, device="auto"):
    """Return diameter (Å), diameter (nm), volume fraction, count, dV/dd (cm³/nm/g)."""
    field = field_from_atoms(
        atoms, radii=1.7, grid_spacing=0.3, probe_radius=1.325,
        device=device, torch_dtype=torch.float64,
    )
    pores = label_pores(
        field.void_mask, voxel_volume=field.grid.voxel_volume, pbc=atoms.pbc,
        min_voxels=int(np.ceil(1.0 / field.grid.voxel_volume)),
    )
    radii = field.clearance.detach().cpu().numpy()
    radii = radii[(pores.labels > 0) & (radii >= 0.5)]
    if not radii.size:
        raise ValueError("No pores meet the analysis thresholds")

    upper = max(30.0, 2 * radii.max())
    result = pore_size_distribution(
        field, method="all_void", weighting="sphere", labels=pores,
        min_radius=0.5, bin_width=0.5, max_diameter=upper,
    )
    _, counts = bin_distribution(radii, 1.0, bin_width=0.5, max_diameter=upper)
    return np.column_stack((
        result.diameters_A, result.diameters_nm,
        result.normalised * result.bin_width_A, counts, result.dV_dd_cc_per_nm_per_g,
    ))
