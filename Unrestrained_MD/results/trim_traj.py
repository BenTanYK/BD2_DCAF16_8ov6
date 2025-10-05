import MDAnalysis as mda
import sys

n_run = int(sys.argv[1])

prmtop  = '../structures/complex.prmtop'
dcd = f"run{n_run}/traj.dcd"
savedir = f"run{n_run}"

u = mda.Universe(prmtop, dcd)
Ntot = len(u.trajectory)
frames_to_keep = int(round(Ntot * 15.0 / 100.0))

with mda.Writer(f"{savedir}/first_15ns.dcd", u.atoms.n_atoms) as W:
    for ts in u.trajectory[:frames_to_keep]:
        W.write(u.atoms)