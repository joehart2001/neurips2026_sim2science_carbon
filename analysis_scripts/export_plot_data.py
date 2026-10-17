#!/usr/bin/env python3
"""Average corrected data with sample SEM; normalize each PSD run before averaging."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'plot_data'

def save(p,cols,header):
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(p,np.column_stack(cols),delimiter=',',header=header,comments='')
def stats(a):
    a=np.asarray(a)
    assert len(a)==5 and np.isfinite(a).all()
    return a.mean(axis=0),a.std(axis=0,ddof=1)/np.sqrt(len(a))
def main():
    rows=json.loads((ROOT/'manifest.json').read_text())['runs']
    for model,comp in sorted({(r['model'],r['composition']) for r in rows}):
        group=sorted([r for r in rows if (r['model'],r['composition'])==(model,comp)],key=lambda r:r['run'])
        arrays=[np.loadtxt(ROOT/r['outputs']['pdf'],delimiter=',',skiprows=1) for r in group]
        x=arrays[0][:,0]
        for a in arrays:np.testing.assert_array_equal(a[:,0],x)
        mean,se=stats([a[:,1:3] for a in arrays])
        save(OUT/model/comp/'pdf.csv',[x,mean[:,0],se[:,0],mean[:,1],se[:,1]],'r_A,g_r_mean,g_r_sem,G_r_mean,G_r_sem')
        counts=[]
        for r in group:
            p=ROOT/r['outputs']['rings'];a=np.loadtxt(p,delimiter=',' if p.suffix=='.csv' else None,skiprows=1 if p.suffix=='.csv' else 0)
            c=a[:,1] if p.suffix=='.csv' else a[:,0]
            if len(c)<14:c=np.pad(c,(0,14-len(c)))
            counts.append(c[3:14]/c.sum()*100)
        mean,se=stats(counts)
        save(OUT/model/comp/'rings.csv',[np.arange(3,14),mean,se],'ring_size,percent_mean,percent_sem')
        arrays=[np.loadtxt(ROOT/r['outputs']['psd'],delimiter=',',skiprows=1) for r in group]
        x=arrays[0][:,1];mask=(x>=.1)&(x<=5);x=x[mask];curves=[]
        for a in arrays:
            if not np.array_equal(a[:,1],arrays[0][:,1]):raise ValueError('PSD bin grids differ')
            y=a[mask,4];area=np.trapezoid(y,x)
            if area<=0:raise ValueError('Invalid PSD area')
            curves.append(y/area)
        mean,se=stats(curves)
        save(OUT/model/comp/'psd.csv',[x*10,mean*.1,se*.1],'diameter_A,psd_mean_per_A,psd_sem_per_A')
        np.testing.assert_allclose(np.trapezoid(mean,x),1)
    print('Exported all 24 consistent curves with sample SEM')
if __name__=='__main__':main()
