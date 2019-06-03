import sys
import numpy as np
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
class ForceFields():
    def __init__(self,force_field):
        #Lennard jones
        #-A*(-12*xi + 12*xj)/((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)**7 + B*(-6*xi + 6*xj)/((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)**4
        pass


class ParticleGroup():
    def __init__(self,coords,velocs,masses,znumbers):
        try:
            assert isinstance(znumbers,np.ndarray) and len(znumbers.shape) == 1
            assert isinstance(masses,np.ndarray) and len(masses.shape) == 1
            assert isinstance(coords,np.ndarray) and len(coords.shape) == 2
            assert isinstance(velocs,np.ndarray) and len(velocs.shape) == 2
            assert len(velocities.shape) == 2
        except AssertionError:
            raise ValueError("Wrong input to ParticleGroup")
        self.initial_coordinates = coords
        self.initial_velocities = velocs
        self.znumbers = znumbers
        self.masses = masses


    @classmethod
    def from_xyz(cls,input_path,distribution=False):
        coords, znumbers = read_coordinates_from_xyz(input_path,chem_dict)
        masses = get_atomic_masses(znumbers,mass_dict)
        if distribution:
            dist_shape,spread = distribution
            velocs = generate_velocities(coordinates,dist_shape,spread)
        else:
            velocs = read_velocities_from_xyz(input_path)
        particle_group = cls(coords,velocs,masses,znumbers)
        return particle_group
    def set_force_field(self,force_field):
        self.ff = force_field

    def update_positions(self):
        '''updates positions according to some algorithm'''

       

def read_velocities_from_xyz(input_path):
    velocs = []
    with open(input_path) as myfile:
        for num, line in enumerate(myfile):
            if num > 1:
                values = line.split()
                if len(values) != 7:
                    raise RuntimeError('Problem with input file. Maybe its'
                            ' missing some velocities?')
                velocs.append(
                        [float(x) for x in values[5:8]])
        velocs = np.array(velocs)
    return velocs


def parse_elinput(elinput,chem_dict):
    znumbers = []
    for el in elinput:
        try:
            znum = int(el)
        except ValueError:
            znumbers.append(chem_dict[el.lower().capitalize()])
        else:
            znumbers.append(znum)
    return np.array(znumbers)


def ang2bohr(x):
    return x*1.88973


def read_coordinates_from_xyz(input_path,chem_dict):
    coordinates = []
    elinput = []
    with open(input_path) as myfile:
        for num, line in enumerate(myfile):
            if num == 0:
                num_atoms = line
            if num == 1: 
                file_comment = line
            if num > 1:
                values = line.split()
                elinput.append(values[0])
                coordinates.append(
                        [ang2bohr(float(x)) for x in values[1:4]])
        coordinates = np.array(coordinates)
        znumbers = parse_elinput(elinput,chem_dict)
    return coordinates, znumbers


def znumbers_to_symbols(znumbers,chem_dict):
    symbols = [list(chem_dict.keys())[list(chem_dict.values()).index(num)] 
            for num in znumbers]
    return np.array(symbols)


def get_atomic_masses(znumbers,mass_dict):
    masses = [mass_dict[z] for z in znumbers]
    return np.array(masses)


def generate_velocities(coordinates,distribution,spread):
    #note that the atomic unit of velocity is a_0 x E_h / hbar = 
    #bohr x hartree /hbar
    if distribution == 'uniform':
        velocities = np.random.uniform(
                -spread/2,spread/2,coordinates.shape)
    if distribution == 'constant':
        velocities = np.full(coordinates.shape,spread,dtype=np.float)
    return velocities


#def run_verlet(coordinates,force_field,timestep,max_timesteps):
#
#
#    accel_k = force_field(coord_k)/m
#    coord_kp = 2*coord_k - coord_km + accel_k*(timestep**2)
#    print(coord_kp)

#the coordinates are obtained in bohr, everything is in atomic units
coordinates, znumbers = read_coordinates_from_xyz(input_path,chem_dict)
symbols = znumbers_to_symbols(znumbers,chem_dict)
masses = get_atomic_masses(znumbers,mass_dict)
velocities = generate_velocities(coordinates,'uniform',10)
particles = ParticleGroup(coordinates,velocities,masses,znumbers)
particles = ParticleGroup.from_xyz(input_path,('uniform',10))
particles = ParticleGroup.from_xyz(input_pathv,distribution=False)
particles = ParticleGroup.from_xyz(input_path,('constant',0))
#particles = ParticleGroup.from_xyz(input_pathvc,distribution=False)
print(particles.initial_coordinates)
print(particles.initial_velocities)
print(particles.znumbers)
print(particles.masses)
