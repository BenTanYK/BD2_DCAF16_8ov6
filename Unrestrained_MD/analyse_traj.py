import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.distances import dist
import pickle
import sys
from tqdm import tqdm
import os

def getDistance(idx1, idx2, u):
    """
    Get the distance between two atoms in a universe.

    Parameters
    ----------
    idx1 : int
        Index of the first atom
    idx2 : int
        Index of the second atom
    u : MDAnalysis.Universe
        The MDA universe containing the atoms and
        trajectory.

    Returns
    -------
    distance : float
        The distance between the two atoms in Angstroms.
    """
    distance = dist(
        mda.AtomGroup([u.atoms[idx1]]),
        mda.AtomGroup([u.atoms[idx2]]),
        box=u.dimensions,
    )[2][0]
    return distance

def closest_residue_to_point(atoms, point):
    """Find the closest residue in a selection of atoms to a given point"""
    residues = atoms.residues
    distances = np.array([np.linalg.norm(res.atoms.center_of_mass() - point) for res in residues])

    # Find the index of the smallest distance
    closest_residue_index = np.argmin(distances)

    # Return the closest residue
    return residues[closest_residue_index], distances[closest_residue_index]

def obtain_CA_idx(u, res_idx):
    """Function to obtain the index of the alpha carbon for a given residue index"""
    
    selection_str = f"protein and resid {res_idx} and name CA"
    
    selected_CA = u.select_atoms(selection_str)

    if len(selected_CA.indices) == 0:
        print('CA not found for the specified residue...')
    
    elif len(selected_CA.indices) > 1:
        print('Multiple CAs found, uh oh...')

    else:  
        return selected_CA.indices[0]
    
def obtain_angle(pos1, pos2, pos3):

    return mda.lib.distances.calc_angles(pos1, pos2, pos3)

def obtain_dihedral(pos1, pos2, pos3, pos4):

    return mda.lib.distances.calc_dihedrals(pos1, pos2, pos3, pos4)

def obtain_RMSD(run_number, res_range=[0,282]):
    u = mda.Universe('structures/complex_DDB1.prmtop', f'results/run{run_number}/traj.dcd')
    protein = u.select_atoms("protein")

    ref = protein
    R_u =rms.RMSD(protein, ref, select=f'backbone and resid {res_range[0]}-{res_range[1]}')
    R_u.run()

    rmsd_u = R_u.rmsd.T #take transpose
    time = rmsd_u[1]/1000
    rmsd= rmsd_u[2]

    return time, rmsd

def save_RMSD(run_number, res_range=[0,282]):
    """
    Save the RMSD of a given run in a .csv file
    """
    time, RMSD = obtain_RMSD(run_number, res_range)

    df = pd.DataFrame()
    df['Time (ns)'] = time
    df['RMSD (Angstrom)'] = RMSD

    filename = 'RMSD.csv'

    df.to_csv(f"results/run{run_number}/{filename}")

    return df

def obtain_RMSF(run_number, res_range=[0,282]):
    u = mda.Universe('structures/complex_DDB1.prmtop', f'results/run{run_number}/traj.dcd')
    
    average = align.AverageStructure(u, u, select='protein and name CA',
                                 ref_frame=0).run()
    
    ref = average.results.universe
    
    average = align.AverageStructure(u, u, select='protein and name CA', ref_frame=0).run()

    aligner = align.AlignTraj(u, ref,
                            select='protein and name CA',
                            in_memory=True).run()

    c_alphas = u.select_atoms(f'protein and name CA and resid {res_range[0]}-{res_range[1]}')
    R = rms.RMSF(c_alphas).run()

    res = c_alphas.resids
    rmsf = R.results.rmsf

    return res, rmsf

def save_RMSF(run_number, res_range=[0,282]):
    """
    Save the RMSD of a given run in a .csv file
    """
    residx, RMSF = obtain_RMSF(run_number, res_range)

    df = pd.DataFrame()
    df['Residue index'] = residx
    df['RMSF (Angstrom)'] = RMSF
    df.to_csv(f"results/run{run_number}/RMSF.csv")

    return df

def calc_RMSD_RMSF(run_numbers):
    """run_numbers is a list of runs to analyse"""
    for n_run in run_numbers:
        # complex
        print(f"\nGenerating RMSD for run {n_run}")
        save_RMSD(n_run)

        # DCAF16
        time, RMSD = obtain_RMSD(n_run, [0,172])
        df = pd.DataFrame()
        df['Time (ns)'] = time
        df['RMSD (Angstrom)'] = RMSD
        filename = 'RMSD_DCAF16.csv'
        df.to_csv(f"results/run{n_run}/{filename}")

        # BD2
        time, RMSD = obtain_RMSD(n_run, [174, 282])
        df = pd.DataFrame()
        df['Time (ns)'] = time
        df['RMSD (Angstrom)'] = RMSD
        filename = 'RMSD_BD2.csv'
        df.to_csv(f"results/run{n_run}/{filename}")

        # complex
        print(f"\nGenerating RMSF for  run {n_run}")
        save_RMSF(n_run)   

        # DCAF16
        residx, RMSF = obtain_RMSF(n_run, [0,172])
        df = pd.DataFrame()
        df['Residue index'] = residx
        df['RMSF (Angstrom)'] = RMSF
        df.to_csv(f"results/run{n_run}/RMSF_DCAF16.csv")

        # BD2
        residx, RMSF = obtain_RMSF(n_run, [174,282])
        df = pd.DataFrame()
        df['Residue index'] = residx
        df['RMSF (Angstrom)'] = RMSF
        df.to_csv(f"results/run{n_run}/RMSF_BD2.csv") 

def obtain_Boresch_dof(u, dof):
    
    rec_group =  [4, 18, 37, 56, 96, 107, 136, 160, 177, 193, 215, 226, 245, 264, 286, 307, 318, 332, 346, 920, 941, 2068, 2084, 2095, 2102, 2112, 2123, 2133, 2140, 2164, 2183, 2197, 2414, 2428, 2442, 2463]
    lig_group =  [3045, 3055, 3065, 3096, 3128, 3211, 3227, 3239, 3255, 3270, 3280, 3299, 3306, 3325, 3342, 3354, 4040, 4061, 4083, 4103, 4115, 4132, 4147, 4163, 4179, 4189]
    
    res_b = 8
    res_c = 142
    res_B = 262
    res_C = 222 

    group_a = u.atoms[rec_group]
    group_b = u.atoms[[obtain_CA_idx(u, res_b)]]
    group_c = u.atoms[[obtain_CA_idx(u, res_c)]]
    group_A = u.atoms[lig_group]
    group_B = u.atoms[[obtain_CA_idx(u, res_B)]]
    group_C = u.atoms[[obtain_CA_idx(u, res_C)]]

    pos_a = group_a.center_of_mass()
    pos_b = group_b.center_of_mass()
    pos_c = group_c.center_of_mass()
    pos_A = group_A.center_of_mass()
    pos_B = group_B.center_of_mass()
    pos_C = group_C.center_of_mass()

    dof_indices = {
        'thetaA' : [pos_b, pos_a, pos_A],
        'thetaB' : [pos_a, pos_A, pos_B],
        'phiA' : [pos_c, pos_b, pos_a, pos_A],
        'phiB': [pos_b, pos_a, pos_A, pos_B],
        'phiC': [pos_a, pos_A, pos_B, pos_C]
    }

    indices = dof_indices[dof]

    if len(indices) == 3:
        return obtain_angle(indices[0], indices[1], indices[2])

    else:
        return obtain_dihedral(indices[0], indices[1], indices[2], indices[3])
 

# calc_RMSD_RMSF([1,2,3])

run_number = int(sys.argv[1])

for dof in ['thetaA', 'thetaB', 'phiA', 'phiB', 'phiC']:
    if os.path.exists(f'results/run{run_number}/{dof}.pkl'):
        continue
    else:
        print(f"Performing Boresch analysis for {dof} run {run_number}")

        u = mda.Universe('structures/complex_DDB1.prmtop', f'results/run{run_number}/traj.dcd')

        vals = []

        for ts in tqdm(u.trajectory, total=u.trajectory.n_frames, desc='Frames analysed'):
            vals.append(obtain_Boresch_dof(u, dof))

        frames = np.arange(1, len(vals) + 1)

        dof_data = {
            'Frames': frames,
            'Time (ns)': np.round(0.01 * frames, 6),
            'DOF values': vals
        }

        # Save interface data to pickle
        file = f'results/run{run_number}/{dof}.pkl'
        with open(file, 'wb') as f:
            pickle.dump(dof_data, f)