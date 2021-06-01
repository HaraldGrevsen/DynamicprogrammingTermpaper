#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:31:19 2021
@author: annesophiesoeandersen
"""
import os
import numpy as np
os.getcwd()
os.chdir("/Users/annesophiesoeandersen/Documents/Dynamic Programming/test")

# load general packages
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# load modules related to this exercise

from model_dc_multidim import model_dc_multidim
import time

t0 = time.time()  # set the starting time
model = model_dc_multidim()
model.setup()
model.create_grids()
model.solve()
model.simulate()
t1 = time.time() # set the ending time
print(f'time: {t1-t0:.8} seconds') # print the total time


# Figure - policy functions
def figure_multi_dim(par,sol,h,i_k):
    k = par.grid_k[i_k]
    if h == 0:
        print(f'Not working')
        ts = [par.T-3, par.T-10, par.T-20, par.T-30, par.T-40]
        print(f'k={k:.3}')
    elif h == 1:
        print(f'Working part-time')
        print(f'k={k:.3}')
        ts = [par.T-1, par.T-10, par.T-20, par.T-30, par.T-40]
    elif h == 2:
        print(f'Working full-time')
        print(f'k={k:.3}')
        ts = [par.T-1, par.T-10, par.T-20, par.T-30, par.T-40]
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    for i in ts:
        ax.scatter(par.grid_m,sol.c[i-1,h,:,i_k], label=f't = {i}')
    ax.set_xlabel(f"$m_t$")
    ax.set_ylabel(f"$c(m_t,h_{{t}} = {h}, k = {k:.3})$")
    ax.set_xlim([0,3])
    ax.set_ylim([0,3])
    ax.set_title(f'Consumption function')
    plt.legend()
    plt.show()

figure_multi_dim(model.par,model.sol,0,2)
figure_multi_dim(model.par,model.sol,1,2)
figure_multi_dim(model.par,model.sol,2,2)

# Figures - simulations

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
ax.plot(np.arange(model.par.simT),np.mean(model.sim.m,1))
ax.set_xlabel(f"age")
ax.set_ylabel(f"Cash-on-hand $m_t$")
ax.set_title(f'Average Cash on hands')
plt.show()

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
ax.plot(np.arange(model.par.simT),np.mean(model.sim.c,1))
ax.set_xlabel(f"age")
ax.set_ylabel(f"Consumption $c_t$")
ax.set_title(f'Average consumption')
plt.show()

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
ax.plot(np.arange(model.par.simT),np.mean(model.sim.h,1))
ax.set_xlabel(f"age")
ax.set_ylabel(f"Labour supply $h_t$")
ax.set_title(f'Average hours worked')
plt.show()

print(model.sim.k)
k = model.sim.k
m = model.sim.m
c = model.sim.c
h = model.sim.h
a = model.sim.a
wage = model.sim.wage
p = model.sim.p
s = model.sim.s

shock1 = model.par.eps_w
shock2 = model.par.eps_ts