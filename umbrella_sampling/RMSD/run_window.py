import openmm as mm
import openmm.app as app
import openmm.unit as unit
import sys
from sys import stdout
import numpy as np
import os

"""Command line arguments"""

RMSD_0 = np.round(float(sys.argv[1]), 3)
jobid = int(sys.argv[2])

"""Read global params from params.in"""

def read_param(param_str, jobid):
    """
    Read in a specific parameter and assign the parameter value to a variable
    """
    with open(f'jobs/{jobid}.in', 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith(param_str):
                parts = line.split(' = ')
                part = parts[1].strip()

    values = part.split(' ')
    value = values[0].strip()

    # Attempt int conversion
    try:
        value = int(value)
    except:
        value = str(value)

    return value

# MD parameters
timestep = read_param('timestep', jobid)
save_traj = read_param('save_traj', jobid)
equil_steps = int(read_param('equilibration_time', jobid)//(timestep*1e-6))
sampling_steps = int(read_param('sampling_time', jobid)//(timestep*1e-6))
record_steps = read_param('n_steps_between_sampling', jobid)

# Force constants
k_RMSD = read_param('k_RMSD', jobid)

# Directory to save all results
run_number = read_param('run_number', jobid)
restraint_type = read_param('RMSD_restraint', jobid)
species = read_param('species', jobid)
savedir = f"results/{restraint_type}_RMSD/{species}/run{run_number}"

if species == 'BD2':
    prmtop_filename= 'BD2.prmtop'
    inpcrd_filename = 'BD2.inpcrd'

elif species == 'DCAF16': 
    prmtop_filename = 'DCAF16.prmtop'
    inpcrd_filename = 'DCAF16.inpcrd'    

elif species in ['DCAF16_only', 'BD2_only', 'DCAF16withBD2', 'BD2withDCAF16']:
    prmtop_filename = 'complex_eq.prmtop'
    inpcrd_filename = 'complex_eq.inpcrd'

else:
    raise FileNotFoundError(f"Select one of the following options for species of interest: DCAF16, BD2, DCAF16_only, BD2_only, DCAF16withBD2, BD2withDCAF16")    

# Check to see if there is an existing file
if os.path.exists(f'{savedir}/{RMSD_0}.txt'): # Check if a CV sample file already exists
    raise FileExistsError(f"A file of CV samples already exists for RMSD_0 = {RMSD_0}")

if not os.path.exists(savedir): # Make save directory if it doesn't yet exist
    os.makedirs(savedir)

"""System setup"""

dt = timestep*unit.femtoseconds 

# Load param and coord files
prmtop = app.AmberPrmtopFile(f'structures/{prmtop_filename}')
inpcrd = app.AmberInpcrdFile(f'structures/{inpcrd_filename}')

system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, hydrogenMass=1.5*unit.amu, constraints=app.HBonds)  
integrator = mm.LangevinMiddleIntegrator(0.0000*unit.kelvin, 1.0000/unit.picosecond, dt)

simulation = app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)

# Add reporters to output data
simulation.reporters.append(app.StateDataReporter(f'{savedir}/{RMSD_0}.csv', 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))
simulation.reporters.append(app.StateDataReporter(stdout, 2500, step=True, time=True, potentialEnergy=True, temperature=True, speed=True))

if save_traj=='True':
    simulation.reporters.append(app.DCDReporter(f'{savedir}/{RMSD_0}.dcd', 1000))

# Minimise energy 
simulation.minimizeEnergy()

"""System heating"""

for i in range(50):
    integrator.setTemperature(6*(i+1)*unit.kelvin)
    simulation.step(1000)

simulation.step(1000)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

"""Selection tuple for restraint type"""

if restraint_type == 'backbone':
    restraint_tuple = ('C', 'N', 'CA')

elif restraint_type == 'CA':
    restraint_tuple = ['CA']

elif restraint_type != 'heavy_atom':
    raise ValueError('Select one of the following restraint type options:  heavy_atom, backbone, CA')

"""RMSD atom selection"""
# Ligand = BD2, receptor = DCAF16

# Make lists of residue indices distinguising between interface and DDB1-binding regions of DCAF16
DCAF16_interface_residx = np.append(np.arange(0,55), np.arange(123, 172))
DCAF16_DDB1_binding_residx = np.arange(71,114)

reference_positions = inpcrd.positions

if species == 'BD2':
    ligand_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(0, 109) and atom.name in restraint_tuple
    ]
 
# Add extra restraint for DDB1-binding residues in DCAF16
elif species == 'DCAF16' or species == 'DCAF16_only':
    receptor_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in DCAF16_interface_residx and atom.name in restraint_tuple
    ]
    DDB1_binding_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in DCAF16_DDB1_binding_residx and atom.name in restraint_tuple
    ]

elif species == 'BD2_only':
    ligand_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(173, 282) and atom.name in restraint_tuple
    ]

else: # BD2withDCAF16 or DCAF16withBD2
    receptor_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in DCAF16_interface_residx and atom.name in restraint_tuple
    ]
    DDB1_binding_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in DCAF16_DDB1_binding_residx and atom.name in restraint_tuple
    ]
    ligand_atoms = [
        atom.index for atom in simulation.topology.atoms()
        if atom.residue.index in range(173, 282) and atom.name in restraint_tuple
    ]

"""Tests to ensure we have the right indices"""

if species in ['DCAF16', 'DCAF16_only', 'DCAF16withBD2']:
    for atom in simulation.topology.atoms():
        if atom.index==receptor_atoms[0] and atom.residue.name!='ASN':
            raise ValueError(f'Incorrect residue selection for DCAF16 - residue N1 is missing')
        if atom.index==receptor_atoms[-1] and atom.residue.name!='LEU':
            raise ValueError(f'Incorrect residue selection for DCAF16 - residue L172 is missing')

if species in ['BD2', 'BD2withDCAF16', 'BD2_only']:
    for atom in simulation.topology.atoms():
        if atom.index==ligand_atoms[0] and atom.residue.name!='SER':
            raise ValueError(f'Incorrect residue selection for BD2 - residue SER174 is missing')
        if atom.index==ligand_atoms[-1] and atom.residue.name!='ASP':
            raise ValueError(f'Incorrect residue selection for BD2 - residue ASP109 is missing')

"""Applying RMSD forces"""

if species in ['BD2', 'BD2_only']:
    ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*(rmsd-rmsd_0)^2')
    ligand_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    ligand_rmsd_force.addGlobalParameter('k_lig', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, ligand_atoms))
    system.addForce(ligand_rmsd_force)

elif species == 'DCAF16' or species == 'DCAF16_only':
    receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*(rmsd-rmsd_0)^2')
    receptor_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    receptor_rmsd_force.addGlobalParameter('k_rec', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
    system.addForce(receptor_rmsd_force)
    simulation.context.reinitialize(preserveState=True)

    DDB1_rmsd_force = mm.CustomCVForce('0.5*k_DDB1*rmsd^2')
    DDB1_rmsd_force.addGlobalParameter('k_DDB1', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
    DDB1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, DDB1_binding_atoms))
    system.addForce(DDB1_rmsd_force)

elif species == 'DCAF16withBD2':
    receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*(rmsd-rmsd_0)^2')
    receptor_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    receptor_rmsd_force.addGlobalParameter('k_rec', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
    system.addForce(receptor_rmsd_force)
    simulation.context.reinitialize(preserveState=True)

    DDB1_rmsd_force = mm.CustomCVForce('0.5*k_DDB1*rmsd^2')
    DDB1_rmsd_force.addGlobalParameter('k_DDB1', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
    DDB1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, DDB1_binding_atoms))
    system.addForce(DDB1_rmsd_force)
    simulation.context.reinitialize(preserveState=True)

    ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*rmsd^2')
    ligand_rmsd_force.addGlobalParameter('k_lig', 30 * unit.kilocalories_per_mole / unit.angstrom**2)
    ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, ligand_atoms))
    system.addForce(ligand_rmsd_force)

else: #BD2withDCAF16
    receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*rmsd^2')
    receptor_rmsd_force.addGlobalParameter('k_rec', 30 * unit.kilocalories_per_mole / unit.angstrom**2)
    receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
    system.addForce(receptor_rmsd_force)
    simulation.context.reinitialize(preserveState=True)

    DDB1_rmsd_force = mm.CustomCVForce('0.5*k_DDB1*rmsd^2')
    DDB1_rmsd_force.addGlobalParameter('k_DDB1', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
    DDB1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, DDB1_binding_atoms))
    system.addForce(DDB1_rmsd_force)
    simulation.context.reinitialize(preserveState=True)

    ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*(rmsd-rmsd_0)^2')
    ligand_rmsd_force.addGlobalParameter('rmsd_0', float(RMSD_0) * unit.angstrom)
    ligand_rmsd_force.addGlobalParameter('k_lig', k_RMSD * unit.kilocalories_per_mole / unit.angstrom**2)
    ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, ligand_atoms))
    system.addForce(ligand_rmsd_force)

simulation.context.reinitialize(preserveState=True)

"""Collecting CV samples"""

print('running window', RMSD_0)

# Equilibration if specified
if equil_steps>0:
    simulation.step(equil_steps) 

# Run the simulation and record the value of the CV.
cv_values=[]

for i in range(sampling_steps//record_steps):

    simulation.step(record_steps)

    if species in ('DCAF16', 'DCAF16_only', 'DCAF16withBD2'):
        current_cv_value = receptor_rmsd_force.getCollectiveVariableValues(simulation.context)

    else: # BD2, BD2, BD2_only, BD2withDCAF16
        current_cv_value = ligand_rmsd_force.getCollectiveVariableValues(simulation.context)    
    
    cv_values.append([i, current_cv_value[0]])

# Final save
np.savetxt(f'{savedir}/{RMSD_0}.txt', np.array(cv_values))

# Delete job parameter file once all calculations are finished
os.remove(f'jobs/{jobid}.in')

print('Completed window', RMSD_0)
