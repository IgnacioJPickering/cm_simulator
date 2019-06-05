import sys
import numpy as np
#input_path = './test_data/some_atoms.xyz'
#input_pathv = './test_data/some_atoms_with_velocs.xyz'
#input_pathvc = './test_data/some_atoms_with_velocs_corrupted.xyz'
mass_dict = {znum:mass for znum,mass in enumerate([0.,1.00811,4.002602,6.997,
    9.0121831,10.821,12.0116,14.00728,15.99977,18.998403,20.1797,22.990,24.305,
    26.982,28.085,30.974,32.06,35.45,39.948])}
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


class VerletPropagator():
    #velocities are calculated as deltax/deltat (order del_t), Stormer-Verlet
    #this is done for simplicity
    def __init__(self,del_t):
        self.del_t = del_t


    def _update_term(self,coords,masses,forces):
        return np.divide(forces,masses[:,np.newaxis])*(self.del_t**2)


    def _propagate_coords_first_ts(self,coords,velocs,masses,forces):
        coords_np1 = coords + velocs*self.del_t + \
            0.5*self._update_term(coords,masses,forces)
        velocs_np1 = (coords_np1-coords)*(1/self.del_t)
        return coords_np1, velocs_np1


    def _propagate_coords_after_first_ts(self,coords,coords_nm1,masses,forces):
        coords_np1 = 2*coords - coords_nm1 + self._update_term(
                coords,masses,forces)
        velocs_np1 = (coords_np1-coords)*(1/self.del_t)
        return coords_np1, velocs_np1


    def propagate(self,particles,time_step):
        forces = particles.get_forces()
        coords = particles.coords
        coords_nm1 = particles.coords_nm1
        velocs = particles.velocs
        masses = particles.masses
        if time_step == 0:
            return self._propagate_coords_first_ts(coords,velocs,masses,forces)
        else:
            return self._propagate_coords_after_first_ts(
                    coords,coords_nm1,masses,forces)


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


    def ff(self,d_vec,r_vec):
        #the force field calculates the total force over 1 particle
        #given a vector of differences and a vector of distances
        #of the neighbohrs
        A = self.A
        B = self.B
        scalar_force = np.divide(12*A,d_vec**14) - np.divide(6*B,d_vec**8)
        vector_force = scalar_force[:,np.newaxis]*r_vec
        return vector_force

class Box():
    def __init__(self,xlo,xhi,ylo,yhi,zlo,zhi):
        #simulation box, particles have boxes attached to them, perhaps its better
        #to do it the other way round
        self.lo = [xlo,ylo,zlo]
        self.hi = [xhi,yhi,zhi]


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
                    coordinates,distribution,target_temperature=tt,
                    masses=masses)
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
        self.rcut = self.potential.rcut


    def attach_propagator(self,propagator):
        #I attach for instance the lennard jones potential to calculate the 
        #energies of the system
        self.propagator = propagator


    def attach_box(self,xlo,xhi,ylo,yhi,zlo,zhi):
        #box has to be an array with 6 elements that determine
        #height, width, etc
        self.box = Box(xlo,xhi,ylo,yhi,zlo,zhi)
        


    def _build_neigh_list(self):
        dij = self._calc_pairwise_dist()
        neigh_matrix = np.asarray(dij < self.rcut)
        #I will build a neighbohr list
        #A neighbohr list is a list of length N, where N is the number of 
        #particles
        #in the nth position my list will have a numpy array with the indices
        #of the Rn neighbohrs of the nth atom.
        neigh_list = []
        for atom in range(dij.shape[0]):
            #the neigh_list is appended the nonzero spots of the matrix that 
            #says whether an atom is a neighbohr or not
            #I extract "itself" from the neighbohr list 
            #(each atom is not its own 
            #neighbohr
            atomic_neigh =  np.nonzero(neigh_matrix[atom,:].astype(bool))[0]
            atomic_neigh = atomic_neigh[atomic_neigh != atom]
            neigh_list.append(atomic_neigh)
        return neigh_list


    def get_forces(self):
        d_mat = self._calc_pairwise_dist()
        r_mat = self._calc_pairwise_diff()
        neigh_list = self._build_neigh_list()
        forces = []
        for i,idx_list in enumerate(neigh_list):
            neigh_dist = d_mat[i,idx_list]
            neigh_diff = r_mat[i,idx_list,:]
            neigh_force = self.potential.ff(neigh_dist,neigh_diff)
            forces.append(np.sum(neigh_force,axis=0))
            #these functions are probably handy for debugging
            #print(i, neigh_dist)
            #print(i, neigh_diff)
            #print(i, neigh_force)
        forces = np.array(forces)
        return forces
        #print(neigh_list)


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


    def update_coords_velocs(self,time_step,boundary_conditions='Open'):
        '''updates positions according to some algorithm,
        bounds are the boundary conditions, I will only support reflecting
        for the time being'''
        #get new coordinates
        coords_np1, velocs_np1 = self.propagator.propagate(self,time_step)
        #update the coordinates
        self.coords_nm1 = self.coords
        self.coords = coords_np1
        self.velocs = velocs_np1

        def exist_oob(indices):
            exist=[]
            for j in range(3):
                exist.append(indices[j].shape[0] != 0)
            return exist

        def get_gt_indices(coordinates,hibounds):
            def oob_gt(matrix,bound):
                return (matrix > bound).nonzero()
            gt_indices = []
            for j in range(3):
                gt_indices.append(oob_gt(coordinates[:,j],hibounds[j])[0])
            return np.array(gt_indices)

        def get_lt_indices(coordinates,lobounds):
            def oob_lt(matrix,bound):
                return (matrix < bound).nonzero()
            lt_indices = []
            for j in range(3):
                lt_indices.append(oob_lt(coordinates[:,j],lobounds[j])[0])
            return np.array(lt_indices)
        
        if boundary_conditions == 'reflecting':
            gt_indices = get_gt_indices(self.coords,self.box.hi)
            lt_indices = get_lt_indices(self.coords,self.box.lo)
            exist_oob_gt = exist_oob(gt_indices)
            exist_oob_lt = exist_oob(lt_indices)
            for j in range(3):
                if exist_oob_gt[j]:
                    self.coords[gt_indices[j],j] = self.box.hi[j]
                if exist_oob_lt[j]:
                    self.coords[lt_indices[j],j] = self.box.lo[j]




       

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

#def run_verlet(coordinates,force_field,timestep,max_timesteps):
#
#
#    accel_k = force_field(coord_k)/m
#    coord_kp = 2*coord_k - coord_km + accel_k*(timestep**2)

#the coordinates are obtained in bohr, everything is in atomic units


