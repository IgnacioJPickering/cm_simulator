import sys
import numpy as np
import pycmech as cm
#from ..pycmech import pycmech
input_path = './test_data/some_atoms.xyz'
input_pathv = './test_data/some_atoms_with_velocs.xyz'
input_pathvc = './test_data/some_atoms_with_velocs_corrupted.xyz'
chem_list = ['X','H','He','Li','Be','B','C','N','O','F','Ne','Na',
        'Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V',
        'Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br',
        'Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag',
        'Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',
        'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
        'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po',
        'At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu']
mass_list = [0.,1.00811,4.002602,6.997,9.0121831,10.821,12.0116,
        14.00728,15.99977,18.998403,20.1797]
mass_dict = {znum:mass for znum,mass in enumerate(mass_list)}
chem_dict = {key:value for value,key in enumerate(chem_list)}
coordinates, znumbers = cm.read_coordinates_from_xyz(input_path,chem_dict)
symbols = cm.znumbers_to_symbols(znumbers,chem_dict)
masses = cm.get_atomic_masses(znumbers,mass_dict)
velocities = cm.generate_velocities(coordinates,'uniform',10)
particles = cm.ParticleGroup(coordinates,velocities,masses,znumbers)
particles = cm.ParticleGroup.from_xyz(input_path,('uniform',10))
particles = cm.ParticleGroup.from_xyz(input_pathv,distribution=False)
particles = cm.ParticleGroup.from_xyz(input_path,('constant',0))
#particles = cm.ParticleGroup.from_xyz(input_pathvc,distribution=False)
print(particles.initial_coordinates)
print(particles.initial_velocities)
print(particles.znumbers)
print(particles.masses)
