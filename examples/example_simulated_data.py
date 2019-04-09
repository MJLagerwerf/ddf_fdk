#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:40:55 2018

@author: lagerwer
"""
import odl
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
pix = 1024
voxels = [pix, pix, pix]

# Pick your phantom
# Options: 'Shepp-Logan', 'Defrise', 'Derenzo', 'Hollow cube', 'Cube', 'Var obj'
phantom = 'Foam'
#lp = '/export/scratch2/lagerwer/NNFDK_results/nTrain_optim_1024_lim_ang/'
#f_load_path = lp + 'CS_f.npy'
#g_load_path = lp + 'CS_A64_g.npy'
noise = None#['Poisson', 2 ** 10]

# The amount of projection angles in the measurements
angles = 2000
ang_freq = 32

# Source to center of rotation radius
src_rad = 59
det_rad = 10.9
pix_size = 0.00748 * 4
# Variables above are expressed in the phyisical size of the measured object

# Noise model
# Options: None,  ['Gaussian', %intensity], ['Poisson', I_0], ['loaded data',
#                    filename]
lp = '/export/scratch2/lagerwer/data/FleXray/pomegranate1_02MAR/processed_data/'
dataset = {'g' : lp + 'g_good_ODL.npy',
               'ground_truth' : lp + 'ground_truth.npy',
                'mask' : lp + 'mask.npy'}
# Create a data object
#data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, det_rad)#, load_data=f_load_path)
data_obj = ddf.real_data(dataset, pix_size, src_rad, det_rad, angles, ang_freq)
# Expansion operator and binnin parameter
expansion_op = 'linear'
bin_param = 2

# %% Create the circular cone beam CT class
case = ddf.CCB_CT(data_obj)#
# Initialize the algorithms (FDK, SIRT)
case.init_algo()
case.init_DDF_FDK()
# %%
case.FDK.do('Shepp-Logan')
case.TFDK.do('optim')
# %% Check convolution
case.FDK.show()
case.TFDK.show()


