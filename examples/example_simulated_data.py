#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:40:55 2018

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
import scipy.interpolate as sp
import gc

pylab.close('all')
t = time.time()
# %% Set variables
# The size of the measured objects in voxels
pix = 256
voxels = [pix, pix, pix]

# Pick your phantom
# Options: 'Shepp-Logan', 'Defrise', 'Derenzo', 'Hollow cube', 'Cube', 'Var obj'
phantom = 'Fourshape'
#lp = '/export/scratch2/lagerwer/NNFDK_results/nTrain_optim_1024_lim_ang/'
#f_load_path = lp + 'CS_f.npy'
#g_load_path = lp + 'CS_A64_g.npy'
noise = None #['Poisson', 2 ** 8]
det_rad = 0
src_rad = 10
angles = 2

for i in range(10):
    data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, det_rad)
    
    
    ### %% Create the circular cone beam CT class
    case = ddf.CCB_CT(data_obj)#
    case.show_phantom()
### Initialize the algorithms (FDK, SIRT)
#case.init_algo()
#case.init_DDF_FDK()
## %%
#
#rec_RL = case.FDK.do('Ram-Lak', compute_results=False)
#rec_HN = case.FDK.do('Hann', compute_results=False)
#CS_RL = rec_RL[:, :, pix // 2]
#CS_HN = rec_HN[:, :, pix //2]
## %%
##xx, yy = np.mgrid[:pix, :pix]
##
##mid = (pix - 1) / 2
##circle = (xx - mid) ** 2 + (yy - mid) ** 2
##max_rad = pix / 2
##
##
##
#def clipNav_annulus(img, circle, max_rad, offset1, offset2): 
#    circle1 = circle < (max_rad * offset2) ** 2
#    circle2 = circle > (max_rad * offset1) ** 2
#    S = np.sum(img * circle1 * circle2)
#    nE = np.sum(np.ones(np.shape(img)) * circle1 * circle2)
#    return S / nE
#
#def MTF(pix, CS):
#    num_bins = pix // 4
#    mid = (pix - 1) / 2
#    xx, yy = np.mgrid[:pix, :pix]
#    circle = (xx - mid) ** 2 + (yy - mid) ** 2
#    max_rad = pix / 2
#    signal = np.zeros(num_bins)
#    for i in range(num_bins):
#        offset1 = (num_bins - i - 1) / num_bins
#        offset2 = (num_bins - i) / num_bins
#        signal[i] = clipNav_annulus(CS, circle, max_rad, offset1, offset2)
#    grad_sig = np.gradient(signal)
#    shift_sig = np.roll(grad_sig, -np.argmax(grad_sig))
#    MTF = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(
#            shift_sig))))[num_bins // 2:] 
#    MTF /= MTF[0]
#    return MTF
#
#
#MTF_RL = MTF(pix, CS_RL)
#MTF_HN = MTF(pix, CS_HN)
#pylab.figure()
#pylab.plot(MTF_RL, label='RL')
#pylab.plot(MTF_HN, label='HN')
#pylab.legend()
## %%    
#case.FDK.do('Ram-Lak')
#case.FDK.show()
#case.FDK.do('Hann')
#case.FDK.show()
#pylab.figure()
