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
phantom = 'Cube'
#lp = '/export/scratch2/lagerwer/NNFDK_results/nTrain_optim_1024_lim_ang/'
#f_load_path = lp + 'CS_f.npy'
#g_load_path = lp + 'CS_A64_g.npy'
noise = None #['Poisson', 2 ** 8]
det_rad = 0
src_rad = 10
angles = 500

data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, det_rad)

## %% Create the circular cone beam CT class
case = ddf.CCB_CT(data_obj)#
## Initialize the algorithms (FDK, SIRT)
case.init_algo()
case.init_DDF_FDK()
# %%

rec_FDK = case.FDK.do('Ram-Lak', compute_results=False)
rec_TFDK = case.TFDK.do(lam=0, compute_results=False)

case.TFDK.do(lam='optim')

# %% Show results
pylab.close('all')
case.table()
case.FDK.show()
case.TFDK.show()
case.show_phantom()

# %%
(data_obj.f - rec_FDK).show()
(data_obj.f - rec_TFDK).show()
# %%    
#pylab.close('all')
#pylab.figure()
