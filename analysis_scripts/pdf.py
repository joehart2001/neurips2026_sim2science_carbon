"""Unweighted pair distribution function for panel (c)."""

from pathlib import Path

import MDAnalysis as mda
import numpy as np
from ase.io import read
from curdf import rdf


def calculate_pdf(filename, device="cpu"):
    """Return r (Å), g(r) and G(r) (Å⁻²) from a LAMMPS data or extxyz file."""
    if Path(filename).suffix == ".data":
        atoms = read(filename, format="lammps-data", atom_style="atomic", read_image_flags=False)
        structure = mda.Universe(filename, atom_style="id type x y z")
        structure.add_TopologyAttr("name", ["C"] * len(atoms))
    else:
        atoms = read(filename)
        structure = atoms.copy()
        structure.set_chemical_symbols(["C"] * len(atoms))
    r, g = rdf(
        structure, species_a="C", species_b="C",
        r_min=0.0, r_max=6.0, nbins=150,
        method="cell_list", device=device, max_neighbors=40000,
    )
    G = 4 * np.pi * r * (len(atoms) / atoms.get_volume()) * (g - 1)
    return np.column_stack((r, g, G))
