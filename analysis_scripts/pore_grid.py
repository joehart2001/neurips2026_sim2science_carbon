"""Uniform periodic grid and exact periodic nearest-atom distances."""
import numpy as np
from scipy.spatial import cKDTree

def periodic_grid(atoms, spacing=0.3):
    L=atoms.cell.lengths()
    if not np.allclose(atoms.cell.array,np.diag(L)) or not np.all(atoms.pbc):
        raise ValueError('Requires axis-aligned orthorhombic periodic cell')
    n=np.ceil(L/spacing).astype(int)
    steps=L/n
    axes=[(np.arange(count)+0.5)*step for count,step in zip(n,steps)]
    points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    tree=cKDTree(np.mod(atoms.positions,L),boxsize=L)
    distance,_=tree.query(points,k=1,workers=8)
    radii=(distance-1.7).reshape(tuple(n))
    volume=float(np.prod(steps))
    # Downstream helpers only use the scalar spacing cubed for voxel volumes.
    info=dict(zip(('x','y','z'),axes))
    info.update(nx=int(n[0]),ny=int(n[1]),nz=int(n[2]),grid_spacing=float(np.cbrt(volume)),
                spacing_A=steps.tolist(),voxel_volume_A3=volume)
    return radii>=1.325,radii,info
