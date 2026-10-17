"""Numerical routines extracted from /home/jh2536/amorphous-carbon/analysis_scripts/pore_analysis/identify_pores_with_overlap.py. Plotting removed."""
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm
from identify_pores_PBC import _merge_periodic_labels, _classify_pore_openness
def identify_pores_periodic_all_spheres(void_mask, local_pore_radii_grid, grid_spacing, atoms):
    """
    Label connected void regions as individual pores with periodic boundary conditions.
    Same as identify_pores_periodic but prepares for all-sphere analysis.
    """
    nx, ny, nz = void_mask.shape

    # Label the mask
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    labels, n_pores = ndi.label(void_mask, structure=structure)

    # Merge labels that wrap around due to periodic boundaries
    labels = _merge_periodic_labels(labels, nx, ny, nz)

    # Relabel to have consecutive IDs
    unique_labels = np.unique(labels[labels > 0])

    if len(unique_labels) == 0:
        print("No pores identified in the structure.")
        return labels, []

    label_map = {old: new for new, old in enumerate(unique_labels, start=1)}

    relabeled = np.zeros_like(labels)
    for old, new in label_map.items():
        relabeled[labels == old] = new

    labels = relabeled
    n_pores = len(unique_labels)

    voxel_volume = grid_spacing ** 3

    pores = []
    print(f"Analyzing {n_pores} pores with periodic boundaries...")
    for pore_id in tqdm(range(1, n_pores + 1)):
        mask = labels == pore_id
        n_voxels = mask.sum()

        if n_voxels == 0:
            continue

        radii = local_pore_radii_grid[mask]

        volume = n_voxels * voxel_volume
        max_radius = np.max(radii)
        mean_radius = np.mean(radii)

        # Classify if pore is open or closed
        pore_type, spanning_directions = _classify_pore_openness(mask, labels, pore_id, nx, ny, nz)

        pores.append({
            "id": pore_id,
            "n_voxels": int(n_voxels),
            "volume": volume,
            "max_radius": float(max_radius),
            "mean_radius": float(mean_radius),
            "type": pore_type,
            "spanning_directions": spanning_directions,
        })

    return labels, pores

def extract_all_spheres_in_pore(
    labels,
    pore_id,
    local_pore_radii_grid,
    grid_info,
    min_radius=0.5,
):
    """
    Extract ALL spheres from a pore - one sphere per void voxel.
    Unlike pack_spheres_in_pore, this allows overlapping spheres.

    Parameters
    ----------
    labels : np.ndarray
        Labeled pore grid
    pore_id : int
        ID of the pore to extract spheres from
    local_pore_radii_grid : np.ndarray
        Grid of local pore radii
    grid_info : dict
        Grid metadata with x, y, z coordinates
    min_radius : float
        Minimum radius threshold

    Returns
    -------
    centers : np.ndarray
        Array of sphere center positions (N, 3)
    radii : np.ndarray
        Array of sphere radii (N,)
    """
    x = grid_info["x"]
    y = grid_info["y"]
    z = grid_info["z"]

    # Get mask for this pore
    pore_mask = labels == pore_id

    if not np.any(pore_mask):
        return np.array([]), np.array([])

    # Get ALL voxels in this pore that meet minimum radius
    valid_mask = pore_mask & (local_pore_radii_grid >= min_radius)

    # Get coordinates and radii of all valid voxels
    indices = np.where(valid_mask)
    if len(indices[0]) == 0:
        return np.array([]), np.array([])

    radii = local_pore_radii_grid[indices]
    positions = np.array([
        x[indices[0]],
        y[indices[1]],
        z[indices[2]]
    ]).T

    return positions, radii

def extract_all_spheres_all_pores(
    labels,
    local_pore_radii_grid,
    grid_info,
    min_radius=0.5,
    min_pore_volume=1.0,
):
    """
    Extract all spheres from all pores (overlapping allowed).

    Parameters
    ----------
    labels : np.ndarray
        Labeled pore grid
    local_pore_radii_grid : np.ndarray
        Grid of local pore radii
    grid_info : dict
        Grid metadata
    min_radius : float
        Minimum radius threshold for individual spheres
    min_pore_volume : float
        Minimum pore volume to include

    Returns
    -------
    all_spheres : list of dict
        List of pore data with all spheres
    """
    n_pores = labels.max()
    voxel_volume = grid_info["grid_spacing"] ** 3

    all_spheres = []

    print(f"Extracting all spheres from {n_pores} pores...")
    for pore_id in tqdm(range(1, n_pores + 1)):
        pore_mask = labels == pore_id
        pore_volume = pore_mask.sum() * voxel_volume

        if pore_volume < min_pore_volume:
            continue

        centers, radii = extract_all_spheres_in_pore(
            labels,
            pore_id,
            local_pore_radii_grid,
            grid_info,
            min_radius=min_radius,
        )

        if len(centers) > 0:
            all_spheres.append({
                'pore_id': pore_id,
                'centers': centers,
                'radii': radii,
                'n_spheres': len(radii),
                'volume': pore_volume,
            })

    return all_spheres

def compute_psd_from_all_spheres(
    atoms,
    all_spheres,
    bin_width=1,  # Å
    save_path=None,
    show=True,
    return_stats=False,
    title_suffix=None,
):
    """
    Compute PSD from all spheres (overlapping allowed).

    Similar to compute_psd_from_packed_spheres but using all void voxels.
    This gives the true distribution of local pore radii throughout the structure.
    """

    # Gather radii from all spheres
    radii_list = []
    for pore_data in all_spheres:
        rs = pore_data.get("radii", [])
        if len(rs):
            radii_list.append(np.asarray(rs, dtype=float))

    if not radii_list:
        print("No spheres available for PSD.")
        empty = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
                 np.array([]), np.array([]))
        if return_stats:
            return (*empty, {})
        return empty

    all_radii = np.concatenate(radii_list)
    # Hygiene: remove invalid radii
    all_radii = all_radii[np.isfinite(all_radii) & (all_radii > 0)]

    if all_radii.size == 0:
        print("No valid (finite, >0) radii available for PSD.")
        empty = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
                 np.array([]), np.array([]))
        if return_stats:
            return (*empty, {})
        return empty

    diameters_A = 2.0 * all_radii
    volumes_A3 = (4.0 / 3.0) * np.pi * all_radii**3

    # Volume-weighted median diameter
    median_diameter_vol_A = None
    total_vol_A3 = float(volumes_A3.sum())
    if total_vol_A3 > 0:
        sort_idx = np.argsort(diameters_A)
        d_sorted = diameters_A[sort_idx]
        v_sorted = volumes_A3[sort_idx]
        cum_frac = np.cumsum(v_sorted) / total_vol_A3
        idx = int(np.searchsorted(cum_frac, 0.5))
        if idx == 0:
            median_diameter_vol_A = float(d_sorted[0])
        elif idx >= d_sorted.size:
            median_diameter_vol_A = float(d_sorted[-1])
        else:
            f0 = cum_frac[idx - 1]
            f1 = cum_frac[idx]
            d0 = d_sorted[idx - 1]
            d1 = d_sorted[idx]
            if f1 > f0:
                w = (0.5 - f0) / (f1 - f0)
                median_diameter_vol_A = float(d0 + w * (d1 - d0))
            else:
                median_diameter_vol_A = float(d1)

    # Ensure plots cover at least 0-3 nm
    max_diameter_A = float(diameters_A.max())
    target_max_A = max(max_diameter_A, 30.0)  # 30 Å = 3 nm
    max_diameter_nm = target_max_A / 10.0
    xlim_nm = (0.0, max(3.0, max_diameter_nm))

    # Binning
    bins_A = np.arange(0.0, target_max_A + bin_width, bin_width)
    counts_vol_A3, edges_A = np.histogram(diameters_A, bins=bins_A, weights=volumes_A3)
    centers_A = 0.5 * (edges_A[:-1] + edges_A[1:])
    total_vol_A3 = counts_vol_A3.sum()
    vol_frac = counts_vol_A3 / total_vol_A3 if total_vol_A3 > 0 else np.zeros_like(counts_vol_A3, float)

    counts_num, _ = np.histogram(diameters_A, bins=bins_A)
    counts_num_total = counts_num.sum()
    counts_num_frac = counts_num / counts_num_total if counts_num_total > 0 else np.zeros_like(counts_num, float)
    median_diameter_num_A = None
    if counts_num_total > 0:
        bin_cum = np.cumsum(counts_num)
        half = counts_num_total * 0.5
        idx = int(np.searchsorted(bin_cum, half))
        idx = min(idx, centers_A.size - 1)
        median_diameter_num_A = float(centers_A[idx])

    # Adsorption-style dV/dd in cc/nm/g
    Vbin_cc = counts_vol_A3 * 1e-24          # Å^3 -> cm^3
    bin_width_nm = bin_width / 10.0          # Å -> nm
    centers_nm = centers_A / 10.0            # Å -> nm

    mass_g = float(atoms.get_masses().sum()) * 1.66053906660e-24  # amu -> g
    dV_dd_cc_per_nm_per_g = Vbin_cc / bin_width_nm / mass_g if mass_g > 0 else np.zeros_like(Vbin_cc, float)

    # Weighted mean diameter
    avg_diam_weighted_A = float(np.average(diameters_A, weights=volumes_A3))
    avg_diam_num_A = diameters_A.mean()

    stats = {
        "n_spheres": int(all_radii.size),
        "total_sphere_volume_A3": float(volumes_A3.sum()),
        "avg_diameter_weighted_A": float(avg_diam_weighted_A),
        "avg_diameter_num_A": float(avg_diam_num_A),
        "median_diameter_volume_A": median_diameter_vol_A,
        "median_diameter_count_A": median_diameter_num_A,
        "min_diameter_A": float(diameters_A.min()),
        "max_diameter_A": float(diameters_A.max()),
        "atoms_mass_g": mass_g,
    }

    print(
        "PSD (all spheres):"
        f" spheres={stats['n_spheres']},"
        f" avg_weighted_diameter={stats['avg_diameter_weighted_A']:.2f} Å,"
        f" avg_number_diameter={stats['avg_diameter_num_A']:.2f} Å,"
        f" median_volume_diameter={stats['median_diameter_volume_A']:.2f} Å,"
        f" median_count_diameter={stats['median_diameter_count_A']:.2f} Å,"
        f" diameter_range={stats['min_diameter_A']:.2f}-{stats['max_diameter_A']:.2f} Å"
    )

    if return_stats:
        return centers_A, vol_frac, counts_num, centers_nm, dV_dd_cc_per_nm_per_g, all_radii, volumes_A3, stats

    return centers_A, vol_frac, counts_num, centers_nm, dV_dd_cc_per_nm_per_g, all_radii, volumes_A3
