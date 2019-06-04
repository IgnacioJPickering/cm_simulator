import sys
import numpy as np
import pycmech as cm

water_path = './test_data/water.xyz'
input_path = './test_data/some_atoms.xyz'
input_pathv = './test_data/some_atoms_with_velocs.xyz'
input_pathvc = './test_data/some_atoms_with_velocs_corrupted.xyz'
def test_coord_water():
    coordinates, znumbers = cm.read_coordinates_from_xyz(water_path)
    assert np.allclose(coordinates, 
            [[1.00,0.00,0.00],
            [1.00,1.00,5.00],
            [0.00,2.00,7.00]],rtol=1e-08,atol=1e-10)

def test_znum_water():
    coordinates, znumbers = cm.read_coordinates_from_xyz(water_path)
    assert np.array_equal(znumbers,[8,1,1])


def test_symbols():
    symbols = cm.znumbers_to_symbols([8,1,1])
    assert np.array_equal(symbols,['O','H','H'])


def test_masses():
    masses = cm.get_atomic_masses([8,1,1])
    assert np.allclose(masses,[15.99977,1.00811,1.00811],rtol=1e-10,atol=1e-12)


#coordinates, znumbers = cm.read_coordinates_from_xyz(input_path)
#symbols = cm.znumbers_to_symbols(znumbers)
#masses = cm.get_atomic_masses(znumbers)
#velocities = cm.generate_velocities(coordinates,'uniform',10)
#particles = cm.ParticleGroup(coordinates,velocities,masses,znumbers)
#particles = cm.ParticleGroup.from_xyz(input_path,('uniform',10))
#particles = cm.ParticleGroup.from_xyz(input_pathv,velocities='xyz')
#particles = cm.ParticleGroup.from_xyz(input_path,('constant',0))
##particles = cm.ParticleGroup.from_xyz(input_pathvc,distribution=False)
#print(particles.initial_coordinates)
#print(particles.initial_velocities)
#print(particles.znumbers)
#print(particles.masses)
