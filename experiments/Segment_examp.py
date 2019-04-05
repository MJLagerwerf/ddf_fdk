
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:40:55 2018

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
#import real_data_class as RD
import time
import pylab
import scipy.interpolate as sp
pylab.close('all')
t = time.time()
# %% Set variables
# The size of the measured objects in voxels
pix = 512
voxels = [pix, pix, pix]

# Pick your phantom
# Options: 'Shepp-Logan', 'Defrise', 'Derenzo', 'Hollow cube', 'Cube', 'Var obj'
phantom = 'Foam'
#lp = '/export/scratch2/lagerwer/NNFDK_results/nTrain_optim_1024_lim_ang/'
#f_load_path = lp + 'CS_f.npy'
#g_load_path = lp + 'CS_A64_g.npy'
noise = ['Poisson', 2 ** 10]

# The amount of projection angles in the measurements
angles = 360
# Source to center of rotation radius
src_rad = 10
det_rad = 0
# Variables above are expressed in the phyisical size of the measured object

# Noise model
# Options: None,  ['Gaussian', %intensity], ['Poisson', I_0], ['loaded data',
#                    filename]

# Create a data object
data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, det_rad)#, load_data=f_load_path)

# Expansion operator and binnin parameter
expansion_op = 'linear'
bin_param = 2

# %% Create the circular cone beam CT class
case = ddf.CCB_CT(data_obj)#,
##                  load_data=g_load_path)
## Initialize the algorithms (FDK, SIRT)
case.init_algo()
case.init_DDF_FDK()
# %%
rec_RL = case.FDK.do('Ram-Lak', compute_results='no')
rec_RL_B2 = case.FDK.filt_LP('Shepp-Logan', ['Bin', 2], compute_results='no')
rec_RL_B4 = case.FDK.filt_LP('Shepp-Logan', ['Bin', 4], compute_results='no')
rec_RL_G2 = case.FDK.filt_LP('Shepp-Logan', ['Gauss', 2], compute_results='no')
rec_RL_G5 = case.FDK.filt_LP('Shepp-Logan', ['Gauss', 5], compute_results='no')
rec_T = case.TFDK.do(lam=6e-5, compute_results='no')

# %%
seg_GT = (np.asarray(data_obj.f) > 0.239)

# %%
def compute_optimal_thresh(rec, seg_GT):
    thresh_max = np.max(rec)
    result = []

    for i in range(50):
        seg = (rec > thresh_max * (0.2 + 0.008 * i) )
        result += [ddf.image_measures.comp_rSegErr(seg, seg_GT)]
    result = np.array(result)
    thresh_opt = thresh_max * (0.2 + 0.008 * np.argmin(result))
    seg = (rec > thresh_opt)
    return result, result.min(), seg
        
result_T, result_opt_T, seg_T = compute_optimal_thresh(rec_T, seg_GT)
result_RL, result_opt_RL, seg_RL = compute_optimal_thresh(rec_RL, seg_GT)
result_B2, result_opt_B2, seg_B2 = compute_optimal_thresh(rec_RL_B2, seg_GT)
result_B4, result_opt_B4, seg_B4 = compute_optimal_thresh(rec_RL_B4, seg_GT)
result_G2, result_opt_G2, seg_G2 = compute_optimal_thresh(rec_RL_G5, seg_GT)

# %%
mid = pix // 2 


# %%
#prc_wrong_T = aff.image_measures.comp_rSegErr(seg_T, seg_GT)
#prc_wrong_RL = aff.image_measures.comp_rSegErr(seg_RL, seg_GT)
#prc_wrong_RL_B2 = aff.image_measures.comp_rSegErr(seg_RL_B2, seg_GT)
#prc_wrong_RL_G2 = aff.image_measures.comp_rSegErr(seg_RL_G2, seg_GT)
#prc_wrong_RL_G5 = aff.image_measures.comp_rSegErr(seg_RL_G5, seg_GT)
pylab.rc('text', usetex=True)
pylab.rcParams.update({'font.size': 24})
pylab.close('all')
#fig, axes = pylab.subplots(2, 3,  figsize=[36, 18])
#fig.suptitle('$S_{err}=$(Incorrectly classified voxels)/(Total voxels)$\cdot100$')
pylab.figure(figsize=[9, 9])
pylab.imshow(np.rot90(seg_GT[:, mid, :]))
#pylab.title('Ground truth segmentation')
pylab.xlabel('x')
pylab.ylabel('z')
pylab.xticks([], [])
pylab.yticks([], [])
pylab.figure(figsize=[9, 9])
pylab.imshow(np.rot90(seg_RL[:, mid, :]))
#pylab.title('Ram-Lak segmentation, $S_{err}$ = '+ \
#    '{:.3}\%'.format(result_opt_RL * 100))
pylab.xlabel('x')
pylab.ylabel('z')
pylab.xticks([], [])
pylab.yticks([], [])
pylab.figure(figsize=[9, 9])
pylab.imshow(np.rot90(seg_T[:, mid, :]))
#pylab.title('T-FDK segmentation, $S_{err}$ = '+ \
#    '{:.3}\%'.format(result_opt_T * 100))
pylab.xlabel('x')
pylab.ylabel('z')
pylab.xticks([], [])
pylab.yticks([], [])
pylab.figure(figsize=[9, 9])
pylab.imshow(np.rot90(seg_B2[:, mid, :]))
#pylab.title('SL-B2-FDK segmentation, $S_{err}$ = '+ \
#    '{:.3}\%'.format(result_opt_B2 * 100))
pylab.xlabel('x')
pylab.ylabel('z')
pylab.xticks([], [])
pylab.yticks([], [])
pylab.figure(figsize=[9, 9])
pylab.imshow(np.rot90(seg_B4[:, mid, :]))
#pylab.title('SL-B4-FDK segmentation, $S_{err}$ = '+ \
#    '{:.3}\%'.format(result_opt_B4 * 100))
pylab.xlabel('x')
pylab.ylabel('z')
pylab.xticks([], [])
pylab.yticks([], [])

pylab.figure(figsize=[9, 9])
pylab.imshow(np.rot90(seg_G2[:, mid, :]))
#pylab.title('SL-G5-FDK segmentation, $S_{err}$ = '+ \
#    '{:.3}\%'.format(result_opt_G2 * 100))
pylab.xlabel('x')
pylab.ylabel('z')
pylab.xticks([], [])
pylab.yticks([], [])







