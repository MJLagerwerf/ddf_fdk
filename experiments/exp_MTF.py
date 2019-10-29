#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:57:32 2018

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
from sacred import Experiment
from sacred.observers import FileStorageObserver
import gc
import pylab
import os
import time
ddf.import_astra_GPU()

ex = Experiment()
# %%
@ex.config
def cfg():
    it_i = 0
    pix = 1024
    # Specific phantom
    phantom = 'cylinder'
    # Number of angles
    angles = 500
    noise = None

    # Source radius
    src_rad = 10

    loadp = '/export/scratch2/lagerwer/AFFDK_results/resubmission/sim_varied_noise'
    f_load_path = None
    g_load_path = None



    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    specifics = 'MTF'
    

# %%
@ex.capture
def CT(pix, phantom, angles, src_rad, noise, Exp_bin, bin_param, f_load_path,
       g_load_path):
    voxels = [pix, pix, pix]
    det_rad = 0
    data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, 
                               det_rad, samp_fac=1)

  
    # %% Create the circular cone beam CT class
    CT_obj = ddf.CCB_CT(data_obj)

    CT_obj.init_algo()
    CT_obj.init_DDF_FDK(bin_param, Exp_bin)
    return CT_obj

# %%
@ex.capture
def save_and_add_artifact(path, arr):
    np.save(path, arr)
    ex.add_artifact(path)
    
@ex.capture
def save_table(case, WV_path):
    case.table()
    latex_table = open(WV_path + '_latex_table.txt', 'w')
    latex_table.write(case.table_latex)
    latex_table.close()
    ex.add_artifact(WV_path + '_latex_table.txt')

@ex.capture
def log_variables(results, Q, RT):
    Q = np.append(Q, results.Q, axis=0)
    RT = np.append(RT, results.rec_time)
    return Q, RT
    


# %%
@ex.automain
def main(pix, specifics, loadp):
    if not os.path.exists('AFFDK_results'):
        os.makedirs('AFFDK_results')
    t2 = time.time()
    # Create a data object
    case = CT()
    t3 = time.time()
    print(t3 - t2, 'seconds to initialize CT object')
    Q = np.zeros((0, 3))
    RT = np.zeros((0))

    f = 'Ram-Lak'
    rec = case.FDK.do(f, compute_results=False)
    MTF = ddf.MTF(pix, rec[:, :, pix // 2])
    save_and_add_artifact(f'{case.WV_path}MTF_FDKRL.npy', MTF)

    f = 'Shepp-Logan'
    LP_filts = [['Gauss', 8], ['Gauss', 5], ['Bin', 2], ['Bin', 5]]
    
    
    rec = case.FDK.do(f, compute_results=False)
    MTF = ddf.MTF(pix, rec[:, :, pix // 2])
    save_and_add_artifact(f'{case.WV_path}MTF_FDKSL.npy', MTF)

    
    for lp in LP_filts:
        rec = case.FDK.filt_LP(f, lp, compute_results=False)
        MTF = ddf.MTF(pix, rec[:, :, pix // 2])
        if lp[0] == 'Gauss':
            save_and_add_artifact(f'{case.WV_path}MTF_FDKSL_GS{lp[1]}.npy',
                                  MTF)
        elif lp[0] == 'Bin':
            save_and_add_artifact(f'{case.WV_path}MTF_FDKSL_BN{lp[1]}.npy',
                                  MTF)

    print('Finished FDKs')
    
    AtA = np.load(f'{loadp}/1/_AtA.npy')
    Atg = np.load(f'{loadp}/1/_Atg.npy')
    DDC_norm = np.load(f'{loadp}/1/_DDC_norm.npy')
    lam = 0.004
    lamI = lam * DDC_norm * np.identity(np.shape(AtA)[0])
    x = case.Exp_op.domain.element(
            np.linalg.solve(AtA + lamI, Atg))
    rec = case.FDK_bin(x)
    MTF = ddf.MTF(pix, rec[:, :, pix // 2])
    save_and_add_artifact(f'{case.WV_path}MTF_TFDK_NOI1.npy', MTF)
    
    
    AtA = np.load(f'{loadp}/7/_AtA.npy')
    Atg = np.load(f'{loadp}/7/_Atg.npy')
    DDC_norm = np.load(f'{loadp}/7/_DDC_norm.npy')
    lam = 8e-5
    lamI = lam * DDC_norm * np.identity(np.shape(AtA)[0])
    x = case.Exp_op.domain.element(
            np.linalg.solve(AtA + lamI, Atg))
    rec = case.FDK_bin(x)
    MTF = ddf.MTF(pix, rec[:, :, pix // 2])
    save_and_add_artifact(f'{case.WV_path}MTF_TFDK_NOI2.npy', MTF)
    


    print('Finished MR-FDK')

    case = None
    gc.collect()
    return Q