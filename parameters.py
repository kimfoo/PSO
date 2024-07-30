import numpy as np

#=== Parameters ===#
data_file_mol_flat	= "ester_mol_big_box"
data_file_slab_flat	= "replicated_slab"
ff_filepath_morse ="ester_tempo_o1n_FF_modified_dih_morse.txt"
template_output_dir = "./lmp_results/itr_{}"  # for Lammps
test_dir="./results"

# ------ the number in data file ------
at		=	8+5		# Atom Types liq+solid
at_mol	=	8
bt		=	10+3		# Bond Types
bt_mol	=	10
agt		=	18+11	# Angle Types
agt_mol	=	18
dt		=	22		# Dihedral Types
it		=	13		# Improper Types


#--- For Simulation
DeltaEnergy         = 1e-5        # accuracy of PPPM method [kcal/mol]


#--- For Solid

Sol_RandSeed        = 202009      # random seed
Sol_Temperature     = 353.15      # temperature [K]
Sol_DampTemp        = 100.0       # inverse of damp constant used in thermostat [fs]

#--- For Liquid

Liq_Temperature     = 353.15      # temperature [K]
Liq_DampTemp        = 100.0       # inverse of damp constant used in thermostat [fs]

#--- For Equilibration

NumSteps_EQ1          = 0     # number of steps of 1st equilibration
					  
#box region
bxl = -8.72088
bxh	= 17.44176
byl = -15.105
byh	= 30.21
bzl = 0.0
bzh	=	233.26550000 #33.26550000





