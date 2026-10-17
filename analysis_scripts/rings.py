"""Carbon ring statistics for panel (b)."""

import numpy as np
from matscipy.rings import ring_statistics


def calculate_rings(atoms):
    """Return ring size, count and count per carbon atom for an ASE Atoms object."""
    carbon = atoms[np.array(atoms.get_chemical_symbols()) == "C"]
    counts = ring_statistics(carbon, cutoff=1.85, maxlength=14)
    return np.column_stack((np.arange(len(counts)), counts, counts / len(carbon)))
