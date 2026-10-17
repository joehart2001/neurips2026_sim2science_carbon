#!/usr/bin/env python3
"""Recalculate paper analyses: carbon rings, actual-N/V unweighted PDF, periodic-grid PSD."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))

def load_atoms(path,fmt):
    from ase.io import read
    from ase.io.lammpsdata import read_lammps_data
    if fmt=='lammps-data':return read_lammps_data(path,atom_style='atomic',read_image_flags=False)
    return read(path,index=-1)

def rings(atoms,out):
    from matscipy.rings import ring_statistics
    atoms=atoms[[s=='C' for s in atoms.get_chemical_symbols()]]
    counts=ring_statistics(atoms,cutoff=1.85,maxlength=14)
    np.savetxt(out,np.column_stack([np.arange(len(counts)),counts,counts/len(atoms)]),delimiter=',',header='ring_size,total_counts,counts_per_carbon_atom',comments='')

def pdf(atoms,path,fmt,out,device):
    from curdf import rdf
    if fmt=='lammps-data':
        import MDAnalysis as mda
        source=mda.Universe(path,atom_style='id type x y z')
        source.add_TopologyAttr('name',['C']*len(source.atoms))
    else:
        source=atoms.copy();source.set_chemical_symbols(['C']*len(source))
    r,g=rdf(source,species_a='C',species_b='C',r_min=0.0,r_max=6.0,nbins=150,method='cell_list',device=device,max_neighbors=40000)
    G=4*np.pi*r*(len(atoms)/atoms.get_volume())*(g-1)
    np.savetxt(out,np.column_stack([r,g,G]),delimiter=',',header='r_A,g_r,G_r_actual_number_density',comments='')

def psd(atoms,out):
    from pore_grid import periodic_grid
    from identify_pores_with_overlap import identify_pores_periodic_all_spheres,extract_all_spheres_all_pores,compute_psd_from_all_spheres
    void,radii,info=periodic_grid(atoms)
    labels,_=identify_pores_periodic_all_spheres(void,radii,info['grid_spacing'],atoms)
    spheres=extract_all_spheres_all_pores(labels,radii,info,min_radius=0.5,min_pore_volume=1.0)
    x,frac,counts,nm,dv,*_=compute_psd_from_all_spheres(atoms,spheres,bin_width=0.5,show=False,return_stats=True)
    np.savetxt(out,np.column_stack([x,nm,frac,counts,dv]),delimiter=',',header='diameter_center_A,diameter_center_nm,volume_fraction,counts_per_bin,dV_dd_cc_per_nm_per_g',comments='')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurement',choices=['rings','pdf','psd','all'])
    parser.add_argument('--model')
    parser.add_argument('--composition',choices=['pure','OH'])
    parser.add_argument('--run',type=int)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'],help='RDF/PDF device; PSD uses exact CPU periodic nearest-neighbour queries')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,help='Default: ROOT/recomputed_data; use ROOT/raw_data explicitly to replace active results')
    args=parser.parse_args()
    rows=json.loads((args.root/'manifest.json').read_text())['runs']
    rows=[r for r in rows if (not args.model or r['model']==args.model) and (not args.composition or r['composition']==args.composition) and (args.run is None or r['run']==args.run)]
    if not rows:parser.error('No matching structures')
    for row in rows:
        path=args.root/row['structure']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['structure_sha256']:
            raise ValueError(f'Structure checksum mismatch: {path}')
        atoms=load_atoms(path,row['format'])
        out=(args.output or args.root/'recomputed_data')/row['model']/row['composition']/f"run_{row['run']}"
        out.mkdir(parents=True,exist_ok=True)
        for measurement in (['rings','pdf','psd'] if args.measurement=='all' else [args.measurement]):
            target=out/(measurement+'.csv')
            if measurement=='rings':rings(atoms,target)
            elif measurement=='pdf':pdf(atoms,path,row['format'],target,args.device)
            else:psd(atoms,target)
            print(target,flush=True)

if __name__=='__main__':main()
