import sys
import numpy as np
import pycmech as cm
import pytest
import matplotlib.pyplot as plt
from math import isclose

ar_path =  './test_data/argon_for_forces.xyz'
water_path =  './test_data/water.xyz'
h_path =  './test_data/atomic_h.xyz'
eps_argon = cm.joule2menergy(1.65e-21)
sig_argon = 3.4
A_ar =  4*eps_argon*sig_argon**12
B_ar =  4*eps_argon*sig_argon**6
cut = 6.
kbol = (1.38067852e-13*6.022e16)

@pytest.mark.parametrize("var,expected",
        [(1,1.8897261246257702),(2,2*1.8897261246257702)])
def test_ang2bohr(var,expected):
    '''Test conversion from angstroem to bohr'''
    assert cm.ang2bohr(var) == expected


@pytest.mark.parametrize("var,expected",
        [(1,1/1.8897261246257702),(2,2/1.8897261246257702)])
def test_bohr2ang(var,expected):
    '''Test conversion from bohr to angstroem'''
    assert cm.bohr2ang(var) == expected


def test_distance_matrix():
    '''Test that the distance matrix is correctly built'''
    particles = cm.ParticleGroup.from_xyz(water_path,('constant',0.))
    dii  = 1e-10
    d12 = cm.bohr2ang(9.635750380837594)
    d13 = cm.bohr2ang(13.886594274127395)
    d23 = cm.bohr2ang(4.628864759856902)
    dij = particles._calc_pairwise_dist()
    assert np.allclose(
            np.array([[dii,d12,d13],[d12,dii,d23],[d13,d23,dii]]),dij,atol=1e-5,rtol=1e-8)
 

def test_lj_pairlist_wcutoff():
    '''Test that the lj cutoff list of interacting pairs is correctly built 1'''
    d12 = cm.bohr2ang(9.635750380837594)
    d23 = cm.bohr2ang(4.628864759856902)
    lj = cm.LJPotential(A=A_ar,B=B_ar,rcut=cut)
    particles = cm.ParticleGroup.from_xyz(water_path,('constant',0.))
    dij = particles._calc_pairwise_dist()
    pl =  lj._get_pairlist_wcutoff(dij)
    assert np.allclose(np.array([d12,d23]),pl) 


def test_lj_pairlist_wcutoff_len():
    '''Test that the lj cutoff list of interacting pairs is correctly built 2'''
    d12 = cm.bohr2ang(9.635750380837594)
    d23 = cm.bohr2ang(4.628864759856902)
    lj = cm.LJPotential(A=A_ar,B=B_ar,rcut=cut)
    particles = cm.ParticleGroup.from_xyz(water_path,('constant',0.))
    dij = particles._calc_pairwise_dist()
    pl =  lj._get_pairlist_wcutoff(dij)
    assert len(pl) == len([d12,d23])


def test_lj_calculator():
    '''Test that lj calculates the potential energy correctly'''
    d12 = cm.bohr2ang(9.635750380837594)
    d23 = cm.bohr2ang(4.628864759856902)
    v12 =  A_ar*d12**(-12)-B_ar*d12**(-6)
    v23 =  A_ar*d23**(-12)-B_ar*d23**(-6)
    particles = cm.ParticleGroup.from_xyz(water_path,('constant',0.))
    particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
    pot = particles.get_pot_energy()
    assert isclose(v12+v23,pot,rel_tol=1e-8,abs_tol=1e-8)


def test_temperature():
    particles = cm.ParticleGroup.from_xyz(h_path,('constant',1.))
    particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
    temperature = particles.get_temperature()
    an_temp = 1.5*1.00811*10/(3*kbol)
    assert isclose(temperature,an_temp,rel_tol=1e-3,abs_tol=1e-8)

def test_kin():
    particles = cm.ParticleGroup.from_xyz(h_path,('constant',1.))
    particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
    kin = particles.get_kin_energy()
    an_kin = 1.5*1.00811*10
    assert isclose(kin,an_kin,rel_tol=1e-3,abs_tol=1e-8)

particles = cm.ParticleGroup.from_xyz(h_path,('constant',1.))
particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
temperature = particles.get_temperature()

particles = cm.ParticleGroup.from_xyz(water_path,('constant',500.0))
particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
pot = particles.get_pot_energy()
kin = particles.get_kin_energy()
temperature = particles.get_temperature()


particles = cm.ParticleGroup.from_xyz(ar_path,('constant',0.))
particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
#print(particles.get_forces())
particles.coords[1,:] = [5.,0.,0.]
#print(particles.get_forces())
particles.coords[1,:] = [5.9,0.,0.]
#print(particles.get_forces())
particles.coords[1,:] = [3.,0.,0.]
#print(particles.get_forces())
particles.coords[1,:] = [3.5*2**(1/6),0.,0.]
#print(particles.get_forces())
particles.coords[1,:] = [3.5,0.,0.]
particles.attach_propagator(cm.VerletPropagator(0.1))
print(particles.coords)
for j in range(10):
    particles.update_positions(j)
    print(particles.coords)


#print(cm.hartree2kelvin(temperature))
#print(pot,kin,temperature)
