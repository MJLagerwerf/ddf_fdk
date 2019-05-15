#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:24:03 2019

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
import pylab
import gc 

# %%
v = 1024
voxels = [v, v, v]
angles = 360

phantom = 'Cluttered sphere'
# Source to center of rotation radius
src_rad = 10
det_rad = 0

noise = None

# Create a data object
#path_base = '/export/scratch2/lagerwer/AFFDK_results/resubmission/'
path_base = '/export/scratch2/lagerwer/ddf_fdk/experiments/AFFDK_results/'
path1 = path_base + 'sim_varied_noise/1/I0256_' 
path2 = path_base + 'sim_varied_noise/10/I01048576_'

# %%
x1 = ddf.make_filt_coeff_from_path(path1, 0.001)
x2 = ddf.make_filt_coeff_from_path(path2, 1e-6)


# %%

filts = [x1, x2, 'Ram-Lak', 'Shepp-Logan', ['Gauss', 8], ['Gauss', 5],
         ['Bin', 2], ['Bin', 5]]

MTF_list = ddf.MTF_x(filts, voxels, angles, src_rad, det_rad)

np.save(path_base + 'MTF_list', MTF_list)
# %%
pylab.close('all')
pylab.rcParams.update({'font.size': 20})
pylab.rc('text', usetex=True)
fig, (ax2) = pylab.subplots(1, 1, figsize=[13.5, 8])

RM = ['MR filter^{Noi, 1}$','MR filter$^{Noi,2}$', 'Ram-Lak', 'Shepp-Logan', 
        'SL + Gauss$_{\sigma=8}$', 'SL + Gauss$_{\sigma=5}$',
        'SL + Bin$_{N=2}$', 'SL + Bin$_{N=5}$']
clr = ['C3', 'C3', 'gold', 'C2', 'C1', 'C6', 'C9', 'C4']


for i in range(len(RM)):
    if i in [0]:
        ax2.plot(MTF_list[i, :], color=clr[i], label=RM[i], ls='-.')
    elif i in [1]:
        ax2.plot(MTF_list[i, :], color=clr[i], label=RM[i], ls='-')
    elif i in [2, 3, 4, 5, 6, 7]:
        ax2.plot(MTF_list[i, :], color=clr[i], label=RM[i], ls='--')

    
ax2.set_title('MTF$(\omega_x, 0, 0)$')
ax2.set_xlabel('$\omega_x>0$')

pylab.draw()
ax2.legend(loc='center left', bbox_to_anchor=(.85, 0.5))

        
#fig.savefig(path_base + '/figures/MTF_comparison.eps', bbox_inches='tight')
fig.savefig(path_base + '/MTF_comparison.eps', bbox_inches='tight')

        
        
        
