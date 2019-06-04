import sys
import numpy as np
import pycmech as cm
import pytest
import matplotlib.pyplot as plt
water_path =  './test_data/water.xyz'

#sanity check for the lennard jones potential, this should look like the 
#argon lennard jones potential
eps_argon = cm.joule2menergy(1.65e-21)
sig_argon = 3.4
A_argon =  4*eps_argon*sig_argon**12
B_argon =  4*eps_argon*sig_argon**6
print(A_argon,B_argon)
rmin_argon = 2**(1/6.)*sig_argon

lj = cm.LJPotential(A=A_argon,B=B_argon,rcut=8.)

#This plot should look like the argon lennard jones potential
#with a minimum at the vertical line
#and going towards 0 at the horizontal line
dist = np.linspace(3.3,8.,100)
ljval =  lj._lj_for_array(dist)
plt.plot(dist,ljval)
ylo,yhi = plt.gca().get_ylim()
xlo,xhi = plt.gca().get_xlim()
plt.vlines(rmin_argon,*plt.gca().get_ylim())
plt.hlines(0.,*plt.gca().get_xlim(),linestyles='dashed')
plt.gca().set_ylim(ylo,yhi)
plt.gca().set_xlim(xlo,xhi)
plt.show()

