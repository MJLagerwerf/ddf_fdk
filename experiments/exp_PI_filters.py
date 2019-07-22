#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:07:29 2019

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


ex = Experiment()
# %%
@ex.config
def cfg():
    it_i = 0
    pix = 1024
    # Specific phantom
    phantoms = ['Cluttered sphere', 'FORBILD']
    phantom = phantoms[it_i]
    # Number of angles
    angs = [360, 96]
    angles = angs[it_i]
    noise_lvls = [['Poisson', 2 ** 8], None]
    noise = noise_lvls[it_i]

    # Source radius
    src_rad = 10



    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    
    specs = ['NOI', 'SV']
    specifics = specs[it_i]

# %%
@ex.capture
def CT(pix, phantom, angles, src_rad, noise, Exp_bin, bin_param, f_load_path,
       g_load_path):
    voxels = [pix, pix, pix]
    det_rad = 0
    if g_load_path is not None and f_load_path is not None:
        data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, 
                               det_rad, load_data_g=g_load_path, 
                               load_data_f=f_load_path)
    elif g_load_path is not None:
        data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, 
                               det_rad, load_data_g=g_load_path)
    elif f_load_path is not None:
        data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, 
                               det_rad, load_data_f=f_load_path)
    else:
        data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, 
                               det_rad)

  
    # %% Create the circular cone beam CT class
    CT_obj = ddf.CCB_CT(data_obj)

    CT_obj.init_algo()
    CT_obj.init_DDF_FDK(bin_param, Exp_bin)
    return CT_obj

# %%
@ex.automain
def main(specifics):
    if not os.path.exists('AFFDK_results'):
        os.makedirs('AFFDK_results')
    t2 = time.time()
    # Create a data object
    case = CT()
    t3 = time.time()
    print(t3 - t2, 'seconds to initialize CT object')

    Q = np.zeros((2, 3))

    
    np.save(case.WV_path + specifics + '_g.npy', case.g)
    ex.add_artifact(case.WV_path + specifics + '_g.npy')

    
    
    case.TFDK.do(lam='optim')
    np.save(case.WV_path + specifics + '_TFDK_rec.npy',
            case.TFDK.results.rec_axis[-1])
    ex.add_artifact(case.WV_path + specifics + '_TFDK_rec.npy')
    Q[0, :] = case.TFDK.results.Q
    
    case.TFDK.do(lam='optim')
    np.save(case.WV_path + specifics + '_TFDK_rec.npy',
            case.TFDK.results.rec_axis[-1])
    ex.add_artifact(case.WV_path + specifics + '_TFDK_rec.npy')
    Q[1, :] = case.TFDK.results.Q


    np.save(case.WV_path + specifics + '_AtA.npy', case.AtA)
    ex.add_artifact(case.WV_path + specifics + '_AtA.npy')
    np.save(case.WV_path + specifics + '_Atg.npy', case.Atg)
    ex.add_artifact(case.WV_path + specifics + '_Atg.npy')
    np.save(case.WV_path + specifics + '_DDC_norm.npy', case.DDC_norm)
    ex.add_artifact(case.WV_path + specifics + '_DDC_norm.npy')
    
    np.save(case.WV_path + specifics + '_BtB.npy', case.BtB)
    ex.add_artifact(case.WV_path + specifics + '_BtB.npy')
    np.save(case.WV_path + specifics + '_Btf.npy', case.Btf)
    ex.add_artifact(case.WV_path + specifics + '_Btf.npy')
    np.save(case.WV_path + specifics + '_DC_norm.npy', case.DC_norm)
    ex.add_artifact(case.WV_path + specifics + '_DC_norm.npy')

    print('Finished AF-FDK')




    case.table()
    latex_table = open(case.WV_path + specifics + '_latex_table.txt', 'w')
    latex_table.write(case.table_latex)
    latex_table.close()
    ex.add_artifact(case.WV_path + specifics + '_latex_table.txt')


    np.save(case.WV_path + specifics + '_Q.npy', Q)
    ex.add_artifact(case.WV_path + specifics + '_Q.npy')
    case = None
    gc.collect()
    return Q