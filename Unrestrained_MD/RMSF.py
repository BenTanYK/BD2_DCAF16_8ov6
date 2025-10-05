import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
import matplotlib.pyplot as plt
from MDAnalysis.analysis.distances import dist
from tqdm import tqdm
import pandas as pd
import pickle
import red

def obtain_RMSF(run_number, res_range=[0, 392], total_time_ns=100.2):  # Adjust the total time if necessary
    total_frames = 10020
    # Calculate frames per ns
    frames_per_ns = total_frames / total_time_ns

    # Calculate the number of frames for the first 40 ns
    frames_40ns = int(frames_per_ns * 40)

    # Load only the frames needed for the first 40 ns
    u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj_40ns.dcd')

    # Slice the trajectory to select only the first 40 ns
    u.trajectory[:frames_40ns]

    average = align.AverageStructure(u, u, select=f'protein and name CA and resid {res_range[0]}-{res_range[1]}', ref_frame=0).run()
    ref = average.results.universe

    aligner = align.AlignTraj(u, ref, select=f'protein and name CA and resid {res_range[0]}-{res_range[1]}', in_memory=True).run()

    c_alphas = u.select_atoms(f'protein and name CA and resid {res_range[0]}-{res_range[1]}')
    R = rms.RMSF(c_alphas).run()

    res = c_alphas.resids
    rmsf = R.results.rmsf

    return res, rmsf

for n_run in [0,1,2,3,4]:

    total_frames = 10020
    frames_per_ns = total_frames / 100.2
    frames_40ns = int(frames_per_ns * 40)

    u = mda.Universe('structures/complex.prmtop', f'results/run{n_run}/traj.dcd')

    with mda.Writer(f'results/run{n_run}/traj_40ns.dcd', u.atoms.n_atoms) as writer:
        for ts in u.trajectory[:frames_40ns]:
            writer.write(u)
    
    print(f"\nTruncated .dcd file generated!\n")

    print(f"\nGenerating DCAF16 RMSF for  run {n_run}")
    residx, RMSF = obtain_RMSF(n_run, [0, 173])
    df = pd.DataFrame()
    df['Residue index'] = residx
    df['RMSF (Angstrom)'] = RMSF
    df.to_csv(f"results/run{n_run}/RMSF_DCAF16.csv")  

    print(f"\nGenerating BRD4 RMSF for  run {n_run}")
    residx, RMSF = obtain_RMSF(n_run, [174, 392])
    df = pd.DataFrame()
    df['Residue index'] = residx
    df['RMSF (Angstrom)'] = RMSF
    df.to_csv(f"results/run{n_run}/RMSF_BRD4.csv")  