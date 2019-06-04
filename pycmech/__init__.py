import sys
import numpy as np
#input_path = './test_data/some_atoms.xyz'
#input_pathv = './test_data/some_atoms_with_velocs.xyz'
#input_pathvc = './test_data/some_atoms_with_velocs_corrupted.xyz'
mass_dict = {znum:mass for znum,mass in enumerate([0.,1.00811,4.002602,6.997,
    9.0121831,10.821,12.0116,14.00728,15.99977,18.998403,20.1797])}
chem_dict = {key:value for value,key in enumerate(['X','H','He','Li','Be','B',
    'C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc',
    'Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
    'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb',
    'Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy',
    'Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl',
    'Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu'])}
genran = np.random.RandomState(12345678)
kbol_si = 1.38067852e-13
kbol = kbol_si*6.022e16
#units to be used internally:
#Distance: angstrom
#Time: femtosecond
#Mass: grams/mol
#Charge: proton charge
#Temperature: Kelvin 

#Energy = g/mol * (ang/femto)**2

#The output of "ready to read quantities" will be handled by a different
#module altogether (and there will be an input of desired units)


class ForceFields():
    def __init__(self,force_field):
        #Lennard jones
        #the lennard jones should only be calculated for r < r_c (lennard jones 
        #is by default
        #Lennard-Jones/cut
        #first a distance matrix has to be built and then interactions within 
        # acertain range are calculated
        #-A*(-12*xi + 12*xj)/((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)**7 + B*(-6*xi + 6*xj)/((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)**4
        pass

class VerletPropagator():
    def __init__(del_t):
        self.del_t = del_t


    def _update_term(self,coords,masses):
        return np.divide(self.ff(coords),masses)*(self.del_t**2)


    def _propagate_coords_first_ts(coords,masses):
        coords_np1 = coords + velocs*self.del_t + \
            0.5*self._update_term(coords,masses)
        return coord_np1


    def _propagate_coords_after_first_ts(coords,coords_nm1,masses):
        coords_np1 = 2*coords - coords_nm1 + self._update_term(coords,masses)
        return coord_np1

    def propagate(coords,coords_nm1,masses,time_step):
        if time_step == 0:
            return self._propagate_coordinates_first_ts(coords,masses)
        else:
            return self._propagate_coordinates_after_first_ts(
                    coords,coords_nm1,masses)


class LJPotential():
    def __init__(self,A,B,rcut): 
        self.A = A
        self.B = B
        self.rcut = rcut
        #if the Lennard-Jones/cut potential is 
        # 4eps( (sig/r)**12 - (sig/r)**6)  then the params are sig and eps
        # the alternate form is A/(r**12) - B/(r**6) in which case
        # A = 4eps sig**12 , B = 4 eps sig**6


    def _get_pairlist_wcutoff(self,diff_mx):
        #this method calculates the coordinates for a given
        #difference matrix
        upper_displaced_triangle = np.mask_indices(diff_mx.shape[0],np.triu,1)
        pair_list_wcutoff = diff_mx[upper_displaced_triangle]
        pair_list_wcutoff = pair_list_wcutoff[pair_list_wcutoff < self.rcut]
        return pair_list_wcutoff


    def _lj_for_array(self,array):
        repulsive_term =  np.power(array,-6)
        attractive_term = np.power(repulsive_term,2)
        return self.A*attractive_term - self.B*repulsive_term


    def calculate(self,diff_mx):
        pair_list_wcutoff = self._get_pairlist_wcutoff(diff_mx)
        array_potential = self._lj_for_array(pair_list_wcutoff)
        sum_potential = np.sum(array_potential)
        return sum_potential

    @staticmethod
    def associated_ff(coords):
        #this calculates the force given a matrix of coords
        def _force(coords):
            pass
            #f = 
        for j in range(coords.shape[0]):
            #_force(coords[j,:]) = sum()
            pass


class ParticleGroup():
    def __init__(self,coords,velocs,masses,znumbers):
        try:
            assert isinstance(znumbers,np.ndarray) and len(znumbers.shape) == 1
            assert isinstance(masses,np.ndarray) and len(masses.shape) == 1
            assert isinstance(coords,np.ndarray) and len(coords.shape) == 2
            assert isinstance(velocs,np.ndarray) and len(velocs.shape) == 2
            assert len(velocs.shape) == 2
        except AssertionError:
            raise ValueError("Wrong input to ParticleGroup")
        self.initial_coordinates = coords
        self.initial_velocities = velocs
        self.coords = self.initial_coordinates
        self.coords_nm1 = None
        self.velocs = self.initial_velocities
        self.znumbers = znumbers
        self.masses = masses


    @classmethod
    def from_xyz(cls,input_path,velocities='xyz'):
        coordinates, znumbers = read_coordinates_from_xyz(input_path)
        masses = get_atomic_masses(znumbers)
        if velocities != 'xyz' and velocities[0] != 'maxwell':
            distribution,spread = velocities
            velocs = generate_velocities(coordinates,distribution,spread=spread)
        elif velocities[0] == 'maxwell':
            distribution,tt = velocities
            velocs = generate_velocities(
                    coordinates,distribution,target_temperature=tt,masses=masses)
        else:
            velocs = read_velocities_from_xyz(input_path)
        particle_group = cls(coordinates,velocs,masses,znumbers)
        return particle_group

    def _calc_pairwise_diff(self):
        #shape coords is (N,3)
        ri = np.expand_dims(self.coords,axis=1)
        #shape ri is (N,1,3)
        rj = np.expand_dims(self.coords,axis=0)
        #shape rj is (1,N,3)
        #shape rij is (N,N,3)
        #this may be slightly confusing, the way this matrix is defined
        #the element rij[0,1,:] has in it coords[1,:]- coords[0,:]
        rij = ri-rj
        return rij

    def _calc_pairwise_dist(self):
        #the maximum allowed difference is 1e-10, if the difference is smaller
        #its set to that value to avoid underflows, since 1/dij is important
        rij = self._calc_pairwise_diff()
        epsilon = 1e-10
        dij = np.maximum(np.sqrt(np.sum(np.square(rij),axis=-1)),epsilon)
        return dij


    def attach_potential(self,potential):
        #I attach for instance the lennard jones potential to calculate the 
        #energies of the system
        self.potential = potential

    def attach_propagator(self,propagator):
        #I attach for instance the lennard jones potential to calculate the 
        #energies of the system
        self.propagator = propagator


    def set_force_field(self,force_field):
        self.ff = force_field


    def get_pot_energy(self):
        #the pairwise distance matrix is calculated each time it is needed, 
        #it is not saved as an internal state (maybe it should, to be changed 
        #later)
        dij = self._calc_pairwise_dist()
        pot = self.potential.calculate(dij)
        return pot


    def get_kin_energy(self):
        vel_sq_vector = np.sum(np.square(self.velocs),axis=-1)
        kin = 0.5*np.matmul(self.masses,vel_sq_vector)
        return kin


    def get_temperature(self):
        #units of temperature assume kbar = 1
        kin = self.get_kin_energy()
        temperature = kin/(3*kbol)
        return temperature


    def update_positions(self,time_step):
        '''updates positions according to some algorithm'''
        #get new coordinates
        coords_np1 = self.propagator.propagate(
                self.coords,self.coords_nm1,self.masses)
        #update the coordinates
        self.coords_nm1 = self.coords
        self.coords = self.coords_np1
       

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


def parse_elinput(elinput):
    znumbers = []
    for el in elinput:
        try:
            znum = int(el)
        except ValueError:
            znumbers.append(chem_dict[el.lower().capitalize()])
        else:
            znumbers.append(znum)
    return np.array(znumbers)

#def convert_units(x,initial,final):
#    #I define menergy units to be the internal energy units of 
#    internal_dict = {('angstrom','bohr'):1.88973,('angstrom','meter'):1e-10,
#            ('menergy','joule'):,('femtosecond','second'):1e-15,} 


def ang2bohr(x):
    return x/0.529177210903


def bohr2ang(x):
    return x*0.529177210903


def meter2bohr(x):
    return ang2bohr(x*1e10)


def bohr2meter(x):
    return bohr2ang(x)*1e10


def joule2hartree(x):
    return x*2.2937122782963e17


def hartree2joule(x):
    return x/2.2937122782963e17


def hartree2kelvin(x):
    return x/(2.2937122782963e17*1.380649e-23)


def joule2menergy(x):
    return x*6.022e16


def read_coordinates_from_xyz(input_path):
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
                        [float(x) for x in values[1:4]])
        coordinates = np.array(coordinates)
        znumbers = parse_elinput(elinput)
    return coordinates, znumbers


def znumbers_to_symbols(znumbers):
    symbols = [list(chem_dict.keys())[list(chem_dict.values()).index(num)] 
            for num in znumbers]
    return np.array(symbols)


def get_atomic_masses(znumbers):
    masses = [mass_dict[z] for z in znumbers]
    return np.array(masses)


def generate_velocities(coordinates,distribution,spread=None,masses=None,
        target_temperature=None):
    #note that the unit of velocity we will use is angstrom/femtosecond
    if distribution == 'uniform':
        velocities = genran.uniform(
                -spread/2,spread/2,coordinates.shape)
    if distribution == 'constant':
        velocities = np.full(coordinates.shape,spread,dtype=np.float)
    if distribution == 'maxwell':
        #this creates a maxwell velocity distribution at the target 
        #temperature
        variance = np.sqrt(np.divide(1,masses)*kbol*target_temperature)
        velocities = []
        for j in range(coordinates.shape[0]):
            velocities.append(genran.normal(
                loc=0.0,scale=variance[j],size=3))
        velocities = np.array(velocities)
    return velocities


#def run_verlet(coordinates,force_field,timestep,max_timesteps):
#
#
#    accel_k = force_field(coord_k)/m
#    coord_kp = 2*coord_k - coord_km + accel_k*(timestep**2)

#the coordinates are obtained in bohr, everything is in atomic units


