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
path_base = '/export/scratch2/lagerwer/AFFDK_results/resubmission/'

exp = 'MTF/1/'
lp_MTF = ['TFDK_NOI1.npy', 'TFDK_NOI2.npy', 'FDKRL.npy', 'FDKSL.npy',
          'FDKSL_GS8.npy', 'FDKSL_GS5.npy', 'FDKSL_BN2.npy','FDKSL_BN5.npy']

size = 254
x = np.arange(size)
MTF_list = np.zeros((len(lp_MTF), size))
filt = [1/4, 1/2, 1/4]
for i in range(len(lp_MTF)):
    MTF_list[i, :] = np.convolve(np.load(f'{path_base}{exp}MTF_{lp_MTF[i]}'),
            filt, 'valid')
    MTF_list[i, :] /= MTF_list[i, 0] 
#    MTF_list[i, :] = np.load(f'{path_base}{exp}MTF_{lp_MTF[i]}')
#    if i in [2, 3, 6, 7]:
#        MTF = np.load(f'{path_base}MTF_{lp_MTF[i]}')    
#        MTF_list[i, :] = np.poly1d(np.polyfit(x, MTF, deg=6))(x)
# %%
pylab.close('all')
pylab.rcParams.update({'font.size': 20})
pylab.rc('text', usetex=True)
fig, (ax2) = pylab.subplots(1, 1, figsize=[13.5, 8])

RM = ['MR filter$^{NOI, 1}$','MR filter$^{NOI, 2}$', 'Ram-Lak', 'Shepp-Logan', 
        'SL + Gauss$_{\sigma=8}$', 'SL + Gauss$_{\sigma=5}$',
        'SL + Bin$_{N=2}$', 'SL + Bin$_{N=5}$']
clr = ['C3', 'C3', 'gold', 'C2', 'C1', 'C6', 'C9', 'C4']


for i in range(len(RM)):
    if i in [0]:
        ax2.plot(MTF_list[i, :], color=clr[i], label=RM[i], ls='-.', lw=3)
    elif i in [1]:
        ax2.plot(MTF_list[i, :], color=clr[i], label=RM[i], ls='-', lw=3)
    elif i in [2, 3, 4, 5, 6, 7]:
        ax2.plot(MTF_list[i, :], color=clr[i], label=RM[i], ls='--', lw=3)

    
ax2.set_title('MTF$(\omega_x, 0, 0)$')
ax2.set_xlabel('$\omega_x>0$')

pylab.draw()
ax2.legend(loc='center left', bbox_to_anchor=(.85, 0.5))


fig.savefig(path_base + '/figures/MTF_comparison.pdf', bbox_inches='tight')
#fig.savefig(path_base + '/MTF_comparison.eps', bbox_inches='tight')

        
        
        
