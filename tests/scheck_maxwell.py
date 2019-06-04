import sys
import numpy as np
import pycmech as cm
import pytest
import matplotlib.pyplot as plt
import math
from math import isclose

many_path =  './test_data/10k_atoms.xyz'
eps_argon = cm.joule2menergy(1.65e-21)
sig_argon = 3.4
A_ar =  4*eps_argon*sig_argon**12
B_ar =  4*eps_argon*sig_argon**6
cut = 6.
kbol = (1.38067852e-13*6.022e16)

#This is a sanity check for the "maxwell" velocity assignment.
#the velocities should reproduce the maxwell velocity distribution

tt = 200
particles = cm.ParticleGroup.from_xyz(many_path,('maxwell',tt))
mass = cm.get_atomic_masses(particles.znumbers)[0]

#These three plots should be gaussian, with mean zero 
#The histograms should match the theoretical distributions
def maxwell_componentwise_theoretical(x,tt,mass):
    return (mass/(2*math.pi*kbol*tt))**(0.5)*np.exp(-mass*x**2/(2*kbol*tt))
    pass

plt.subplot(311)
n,bins,patches = plt.hist(particles.velocs[:,0],bins=100,density=True,facecolor='g',
        alpha=0.5)
xlo,xhi = plt.xlim()
ylo,yhi = plt.ylim()
x = np.linspace(xlo,xhi,1000)
plt.plot(x,maxwell_componentwise_theoretical(x,tt,mass),color='g')
plt.xlim(xlo,xhi)
plt.ylim(ylo,yhi)

plt.subplot(312)
n,bins,patches = plt.hist(particles.velocs[:,1],bins=100,density=True,facecolor='r',
        alpha=0.5)
x = np.linspace(xlo,xhi,1000)
plt.plot(x,maxwell_componentwise_theoretical(x,tt,mass),color='r')
plt.xlim(xlo,xhi)
plt.ylim(ylo,yhi)

plt.subplot(313)
n,bins,patches = plt.hist(particles.velocs[:,2],bins=100,density=True,facecolor='b',
        alpha=0.5)
x = np.linspace(xlo,xhi,1000)
plt.plot(x,maxwell_componentwise_theoretical(x,tt,mass),color='b')
plt.xlim(xlo,xhi)
plt.ylim(ylo,yhi)

plt.show()
speed = np.sqrt(np.sum(np.square(particles.velocs),axis=1))

#This plot should be the maxwell boltzmann velocity distribution, the 
#theoretical distribution and the histogram should closely match
assert mass == 1.00811
def maxwell_theoretical(x,tt,mass):
    return 4*math.pi*(mass/(2*math.pi*kbol*tt))**(1.5)*x**2*np.exp(-mass*x**2/(2*kbol*tt))
n,bins,patches = plt.hist(speed,bins=100,density=True,facecolor='k',alpha=0.5)
xlo,xhi = plt.xlim()
x = np.linspace(xlo,xhi,1000)
plt.plot(x,maxwell_theoretical(x,tt,mass),color='g')
plt.xlim(0,xhi)
plt.show()
