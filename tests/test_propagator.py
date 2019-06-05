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
ar_mass = 39.948
A_ar =  4*eps_argon*sig_argon**12
B_ar =  4*eps_argon*sig_argon**6
rmin_argon = 2**(1/6.)*sig_argon
cut = 6.
kbol = (1.38067852e-13*6.022e16)

@pytest.mark.parametrize("dist",[0.1,0.5,1.,2.,3.,4.,5.,5.5])
def test_forces_magnitude(dist):
    particles = cm.ParticleGroup.from_xyz(ar_path,('constant',0.))
    particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
    particles.coords[1,:] = [dist,0.,0.]
    #forces for these coordinates should be, in magnitude:
    myforce = abs(12*A_ar/(dist**13)-6*B_ar/(dist**7))
    assert isclose(abs(particles.get_forces()[0][0]),
            myforce,rel_tol=1e-8,abs_tol=1e-8) 


@pytest.mark.parametrize("dist",[0.1,0.5,1.,2.,3.,4.,5.,5.5])
def test_forces_newton_third_pair(dist):
    particles = cm.ParticleGroup.from_xyz(ar_path,('constant',0.))
    particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
    particles.coords[1,:] = [dist,0.,0.]
    #forces for these coordinates should be, in magnitude:
    assert isclose(abs(particles.get_forces()[0][0]),
            abs(particles.get_forces()[1][0]),rel_tol=1e-8,abs_tol=1e-8) 


@pytest.mark.parametrize("dist",np.linspace(0.1,3.8,10).tolist())
def test_forces_sign_close(dist):
    particles = cm.ParticleGroup.from_xyz(ar_path,('constant',0.))
    particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
    particles.coords[1,:] = [dist,0.,0.]
    #forces for these coordinates should be, in magnitude:
    assert particles.get_forces()[0][0] < 0. and \
        particles.get_forces()[1][0] > 0.

@pytest.mark.parametrize("dist",np.linspace(3.9,5.5,10).tolist())
def test_forces_sign_far(dist):
    particles = cm.ParticleGroup.from_xyz(ar_path,('constant',0.))
    particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
    particles.coords[1,:] = [dist,0.,0.]
    #forces for these coordinates should be, in magnitude:
    assert particles.get_forces()[0][0] > 0. and \
        particles.get_forces()[1][0] < 0.

def test_forces_eq():
    particles = cm.ParticleGroup.from_xyz(ar_path,('constant',0.))
    particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
    particles.coords[1,:] = [rmin_argon,0.,0.]
    #forces for these coordinates should be, in magnitude:
    assert isclose(particles.get_forces()[0][0],0.,
            rel_tol=1e-8,abs_tol=1e-8) \
            and isclose(particles.get_forces()[1][0],0.,
                    rel_tol=1e-8,abs_tol=1e-8)


def test_propagate_first_ts():
    ts = 5.
    particles = cm.ParticleGroup.from_xyz(ar_path,('constant',0.))
    particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
    particles.coords[1,:] = [4.,0.,0.]
    particles.attach_propagator(cm.VerletPropagator(ts))
    myforce = 12*A_ar/(4.**13)-6*B_ar/(4.**7)
    upterm = (myforce/ar_mass)*ts**2
    new_coord = 4. + 0.5*upterm
    particles.update_coords_velocs(0)
    assert isclose(new_coord,particles.coords[1][0],rel_tol=1e-8,abs_tol=1e-8)

def test_propagate_second_ts():
    ts = 5.
    particles = cm.ParticleGroup.from_xyz(ar_path,('constant',0.))
    particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
    particles.coords[1,:] = [4.,0.,0.]
    particles.attach_propagator(cm.VerletPropagator(ts))
    myforce = 12*A_ar/(4.**13)-6*B_ar/(4.**7)
    upterm = (myforce/ar_mass)*ts**2
    new_coord = 4. + 0.5*upterm
    myforce = 12*A_ar/(new_coord**13)-6*B_ar/(new_coord**7)
    upterm = (myforce/ar_mass)*ts**2
    new_coord = 2*new_coord - 4. + upterm
    particles.update_coords_velocs(0)
    particles.update_coords_velocs(1)
    assert isclose(new_coord,particles.coords[1][0],rel_tol=1e-8,abs_tol=1e-8)

def test_propagate_third_ts():
    ts = 5.
    particles = cm.ParticleGroup.from_xyz(ar_path,('constant',0.))
    particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
    particles.coords[1,:] = [4.,0.,0.]
    particles.attach_propagator(cm.VerletPropagator(ts))
    myforce = 12*A_ar/(4.**13)-6*B_ar/(4.**7)
    upterm = (myforce/ar_mass)*ts**2
    new_coord1 = 4. + 0.5*upterm

    myforce = 12*A_ar/(new_coord1**13)-6*B_ar/(new_coord1**7)
    upterm = (myforce/ar_mass)*ts**2
    new_coord = 2*new_coord1 - 4. + upterm

    myforce = 12*A_ar/(new_coord**13)-6*B_ar/(new_coord**7)
    upterm = (myforce/ar_mass)*ts**2
    new_coord = 2*new_coord - new_coord1 + upterm
    particles.update_coords_velocs(0)
    particles.update_coords_velocs(1)
    particles.update_coords_velocs(2)
    assert isclose(new_coord,particles.coords[1][0],rel_tol=1e-8,abs_tol=1e-8)

particles = cm.ParticleGroup.from_xyz(ar_path,('constant',0.))
particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
particles.coords[1,:] = [4.,0.,0.]
particles.attach_propagator(cm.VelVerletPropagator(5.))
print(particles.coords)
print(particles.velocs)
particles.update_coords_velocs(0)
print(particles.coords)
print(particles.velocs)
particles.update_coords_velocs(1)
print(particles.coords)
print(particles.velocs)

