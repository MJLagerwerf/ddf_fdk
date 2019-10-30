#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:22:31 2019

@author: lagerwer
"""


import odl
import numpy as np
import ddf_fdk as ddf
import astra
ddf.import_astra_GPU()
#import real_data_class as RD
import time
import pylab
import gc
import os
from tqdm import tqdm
pylab.close('all')
t = time.time()




# %% Set variables
# The size of the measured objects in voxels
pix = 256
voxels = [pix, pix, pix]

# Pick your phantom
# Options: 'Shepp-Logan', 'Defrise', 'Derenzo', 'Hollow cube', 'Cube', 'Var obj'
phantom = 'cylinder'
#lp = '/export/scratch2/lagerwer/NNFDK_results/nTrain_optim_1024_lim_ang/'
#f_load_path = lp + 'CS_f.npy'
#g_load_path = lp + 'CS_A64_g.npy'
noise = ['Poisson', 2 ** 8]
det_rad = 0
src_rad = 10
angles = 500
nTests = 500

# %%
data_obj = ddf.phantom(voxels, 'FORBILD', angles, noise, src_rad, det_rad,
                       samp_fac=1)

## %% Create the circular cone beam CT class
case = ddf.CCB_CT(data_obj)#
## Initialize the algorithms (FDK, SIRT)
case.init_algo()
case.init_DDF_FDK()
case.TFDK.do(lam='optim')
x_FB = case.TFDK.results.var[-1] * case.filter_space.weighting.const

# %%
CS_T = np.zeros((nTests, pix))
CS_FB = np.zeros((nTests, pix))
CS_RL = np.zeros((nTests, pix))
CS_G8 = np.zeros((nTests, pix))
CS_B5 = np.zeros((nTests, pix))

LS_T = np.zeros((nTests, pix))
LS_FB = np.zeros((nTests, pix))
LS_RL = np.zeros((nTests, pix))
LS_G8 = np.zeros((nTests, pix))
LS_B5 = np.zeros((nTests, pix))
# %%
def add_results(CS, LS, rec, it):
    mid = np.shape(rec)[0] // 2
    CS[it, :] = rec[:, mid, mid]
    LS[it, :] = rec[mid, mid, :]

def save_results(av_rec, CS, LS, path, meth):
    np.save(f'{path}_rec_{meth}', av_rec)
    np.save(f'{path}_CS_{meth}', CS)
    np.save(f'{path}_LS_{meth}', LS)


# %%
for i in tqdm(range(nTests)):
    data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, det_rad,
                           samp_fac=1)
    
    ## %% Create the circular cone beam CT class
    case = ddf.CCB_CT(data_obj)#
    ## Initialize the algorithms (FDK, SIRT)
    case.init_algo()
    case.init_DDF_FDK()
    case.TFDK.do(lam='optim')
    x = case.TFDK.results.var[-1]
    if i == 0:
        rec_RL = case.FDK.do('Ram-Lak', compute_results=False)
        rec_G8 = case.FDK.filt_LP('Shepp-Logan', ['Gauss', 8],
                               compute_results=False)
        rec_B5 = case.FDK.filt_LP('Shepp-Logan', ['Bin', 5],
                               compute_results=False)
        rec_T = case.FDK_bin(x)
        x_FB /= case.filter_space.weighting.const
        rec_FB = case.FDK_bin(x_FB)
        add_results(CS_RL, LS_RL, rec_RL, i)
        add_results(CS_G8, LS_G8, rec_G8, i)
        add_results(CS_B5, LS_B5, rec_B5, i)
        add_results(CS_T, LS_T, rec_T, i)
        add_results(CS_FB, LS_FB, rec_FB, i)

    else:
        rec = case.FDK.do('Ram-Lak', compute_results=False)
        add_results(CS_RL, LS_RL, rec, i)
        rec_RL += rec
        
        rec = case.FDK.filt_LP('Shepp-Logan', ['Gauss', 8],
                               compute_results=False)
        add_results(CS_G8, LS_G8, rec, i)
        rec_G8 += rec
        
        rec = case.FDK.filt_LP('Shepp-Logan', ['Bin', 5],
                               compute_results=False)
        add_results(CS_B5, LS_B5, rec, i)
        rec_B5 += rec    
        
        rec = case.FDK_bin(x)
        add_results(CS_T, LS_T, rec, i)
        rec_T += rec
        
        rec = case.FDK_bin(x_FB)
        add_results(CS_FB, LS_FB, rec, i)
        rec_FB += rec

rec_RL /= nTests
rec_G8 /= nTests
rec_B5 /= nTests
rec_T /= nTests
rec_FB /= nTests
    

# %%
def compute_sd(CS, CS_av):
    S = np.zeros(np.shape(CS)[1])
    for i in range(np.shape(CS)[0]):
        S += (CS[i, :] - CS_av) ** 2
    return np.sqrt(S / np.shape(CS)[0])

# %%
def plot_slice(GT, av_recs, sd_recs, meths, path=None):
    pylab.figure()
    pylab.plot(GT, label='GT', lw=4)
    x = np.arange(len(GT))
    
    for i in range(len(meths)):
#        pylab.plot(av_recs[i], label=meths[i], lw=2)
        pylab.errorbar(x, av_recs[i], sd_recs[i], label=meths[i], lw=2,
                        errorevery=20, capsize=5)
    pylab.legend()

# %%  
pylab.close('all')
meths = ['RL', 'G8', 'B5', 'T', 'FB']
# %%
GT_CS = data_obj.f[:, pix // 2, pix // 2]
av_recs_CS = [rec_RL[:, pix // 2, pix //2], rec_G8[:, pix // 2, pix //2],
           rec_B5[:, pix // 2, pix //2], rec_T[:, pix // 2, pix //2],
           rec_FB[:, pix // 2, pix //2]]
sd_recs_CS = [compute_sd(CS_RL, rec_RL[:, pix // 2, pix //2]), 
           compute_sd(CS_G8, rec_G8[:, pix // 2, pix //2]),
           compute_sd(CS_B5, rec_B5[:, pix // 2, pix //2]),
           compute_sd(CS_T, rec_T[:, pix // 2, pix //2]),
           compute_sd(CS_FB, rec_FB[:, pix // 2, pix //2])]

plot_slice(GT_CS, av_recs_CS, sd_recs_CS, meths)

# %%
GT_LS = data_obj.f[pix // 2, pix // 2, :]
av_recs_LS = [rec_RL[pix // 2, pix //2, :], rec_G8[pix // 2, pix //2, :],
           rec_B5[pix // 2, pix //2, :], rec_T[pix // 2, pix //2, :],
           rec_FB[pix // 2, pix //2, :]]
sd_recs_LS = [compute_sd(CS_RL, rec_RL[pix // 2, pix //2, :]), 
           compute_sd(CS_G8, rec_G8[pix // 2, pix //2, :]),
           compute_sd(CS_B5, rec_B5[pix // 2, pix //2, :]),
           compute_sd(CS_T, rec_T[pix // 2, pix //2, :]),
           compute_sd(CS_FB, rec_FB[pix // 2, pix //2, :])]

plot_slice(GT_LS, av_recs_LS, sd_recs_LS, meths)

# %%
path = f'/export/scratch2/lagerwer/AFFDK_results/resubmission/bias/I0{noise[1]}/'
if not os.path.exists(path):
    os.makedirs(path)
save_results(rec_RL, CS_RL, LS_RL, path, 'RL')
save_results(rec_G8, CS_G8, LS_G8, path, 'G8')
save_results(rec_B5, CS_B5, LS_B5, path, 'B5')
save_results(rec_T, CS_T, LS_T, path, 'T')
save_results(rec_FB, CS_FB, LS_FB, path, 'FB')


# %%

    