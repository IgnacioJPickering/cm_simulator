import sys
import numpy as np
import pycmech as cm
import pytest
import matplotlib.pyplot as plt
water_path =  './test_data/water.xyz'
ar_path =  './test_data/argon_for_forces.xyz'

#sanity check for the lennard jones potential, this should look like the 
#argon lennard jones potential
eps_argon = cm.joule2menergy(1.65e-21)
sig_argon = 3.4
A_ar =  4*eps_argon*sig_argon**12
B_ar =  4*eps_argon*sig_argon**6
rmin_argon = 2**(1/6.)*sig_argon
print(rmin_argon)
cut = 8.
lj = cm.LJPotential(A=A_ar,B=B_ar,rcut=8.)

#This plot should look like the argon lennard jones potential
#with a minimum at the vertical line
#and going towards 0 at the horizontal line
dist = np.linspace(3.3,8.,100)
ljval =  lj._lj_for_array(dist)
plt.subplot(211)
plt.plot(dist,ljval)
ylo,yhi = plt.gca().get_ylim()
xlo,xhi = plt.gca().get_xlim()
plt.vlines(rmin_argon,*plt.gca().get_ylim())
plt.hlines(0.,*plt.gca().get_xlim(),linestyles='dashed')
plt.gca().set_ylim(ylo,yhi)
plt.gca().set_xlim(xlo,xhi)
#plt.show()


#The idea here is that the force graph should be consistent, in the
#sense that the force should be zero when both argon atoms are 
#at a distance corresponding to the min of potential energy
particles = cm.ParticleGroup.from_xyz(ar_path,('constant',1.))
particles.attach_potential(cm.LJPotential(A=A_ar,B=B_ar,rcut=cut))
#dist = np.linspace(3.3,8.,100)
dist = np.linspace(3.3,8.,100)
force = []
for l in dist:
    particles.coords[1,:] = [l,0.,0.]
    force.append(particles.get_forces()[0,0])
    #print(force)
plt.subplot(212)
plt.plot(dist,force)
ylo,yhi = plt.gca().get_ylim()
xlo,xhi = plt.gca().get_xlim()
plt.vlines(rmin_argon,*plt.gca().get_ylim())
plt.hlines(0.,*plt.gca().get_xlim(),linestyles='dashed')
plt.gca().set_ylim(ylo,yhi)
plt.gca().set_xlim(xlo,xhi)
plt.show()
