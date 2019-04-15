#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:14:05 2019

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
    phantom = 'Foam'
    # Number of angles
    angles = 360
    I0 = [2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14]
    noise = ['Poisson', I0[it_i]]

    # Source radius
    src_rad = 10

    lp = '/export/scratch2/lagerwer/NNFDK/phantoms/'
    f_load_path = None
    g_load_path = None
    
    nTest = 25
    window = [0.2, 0.6]
    
    bin_size = 5
    number_bins = 10

    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    specifics = 'Task_I0' + str(I0[it_i])
    

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
def main(specifics, nTest, window, bin_size, number_bins):
    if not os.path.exists('AFFDK_results'):
        os.makedirs('AFFDK_results')
    t2 = time.time()
    # Create a data object
    case = CT()
    t3 = time.time()
    print(t3 - t2, 'seconds to initialize CT object')

    seg_err = np.zeros((11))
    seg_err_list = np.zeros((11, nTest))
    pore_dist = np.zeros((12, number_bins))
    
    np.save(case.WV_path + specifics + '_g.npy', case.g)
    ex.add_artifact(case.WV_path + specifics + '_g.npy')
    
    # Create the ground truth segementation
    seg_GT = (np.asarray(case.phantom.f) > 0.239)
    np.save(case.WV_path + specifics + '_GT_seg_full.npy', (seg_GT))
    ex.add_artifact(case.WV_path + specifics + '_GT_seg_full.npy')
    np.save(case.WV_path + specifics + '_GT_seg.npy', ddf.get_axis(seg_GT))
    ex.add_artifact(case.WV_path + specifics + '_GT_seg.npy')
    pore_dist[0, :] = ddf.pore_size_distr(seg_GT, bin_size, number_bins)
    
    
    
    
    f = 'Shepp-Logan'
    LP_filts = [['Gauss', 8], ['Gauss', 5], ['Bin', 2], ['Bin', 5]]
    
    rec = case.FDK.do(f, compute_results='no')
    seg_err_list[0, :], seg_err[0], seg = ddf.comp_global_seg(rec, seg_GT,
                nTest, window)
    pore_dist[1, :] = ddf.pore_size_distr(seg, bin_size, number_bins)
    np.save(case.WV_path + specifics + '_FDKSL_seg_full.npy', (seg))
    ex.add_artifact(case.WV_path + specifics + '_FDKSL_seg_full.npy')
    np.save(case.WV_path + specifics + '_FDKSL_seg.npy', ddf.get_axis(seg))
    ex.add_artifact(case.WV_path + specifics + '_FDKSL_seg.npy')


        
    tel = 1
    for lp in LP_filts:
        rec = case.FDK.filt_LP(f, lp, compute_results='no')
        seg_err_list[tel, :], seg_err[tel], seg = ddf.comp_global_seg(rec,seg_GT,
                    nTest, window)
        pore_dist[tel + 1, :] = ddf.pore_size_distr(seg, bin_size, number_bins)
        if lp[0] == 'Gauss':
            np.save(case.WV_path + specifics + '_FDKSL_GS' + str(lp[1]) + 
                    '_seg_full.npy', (seg))
            ex.add_artifact(case.WV_path + specifics + '_FDKSL_GS' + str(lp[1])
                            + '_seg_full.npy')
            np.save(case.WV_path + specifics + '_FDKSL_GS' + str(lp[1]) + 
                    '_seg.npy', ddf.get_axis(seg))
            ex.add_artifact(case.WV_path + specifics + '_FDKSL_GS' + str(lp[1])
                            + '_seg.npy')
        elif lp[0] == 'Bin':
            np.save(case.WV_path + specifics + '_FDKSL_BN' + str(lp[1]) + 
                    '_seg.npy', ddf.get_axis(seg))
            ex.add_artifact(case.WV_path + specifics + '_FDKSL_BN' + str(lp[1])
                            + '_seg.npy')
            np.save(case.WV_path + specifics + '_FDKSL_BN' + str(lp[1]) + 
                    '_seg.npy', ddf.get_axis(seg))
            ex.add_artifact(case.WV_path + specifics + '_FDKSL_BN' + str(lp[1])
                            + '_seg.npy')
        tel += 1


    print('Finished FDKs')
    

    
    
    rec = case.TFDK.do(lam='optim', compute_results='no')
    seg_err_list[6, :], seg_err[6], seg = ddf.comp_global_seg(rec, seg_GT,
                nTest, window)
    pore_dist[7, :] = ddf.pore_size_distr(seg, bin_size, number_bins)
    np.save(case.WV_path + specifics + '_TFDK_seg_full.npy', (seg))
    ex.add_artifact(case.WV_path + specifics + '_TFDK_seg_full.npy')
    np.save(case.WV_path + specifics + '_TFDK_seg.npy', ddf.get_axis(seg))
    ex.add_artifact(case.WV_path + specifics + '_TFDK_seg.npy')

    


    np.save(case.WV_path + specifics + '_AtA.npy', case.AtA)
    ex.add_artifact(case.WV_path + specifics + '_AtA.npy')
    np.save(case.WV_path + specifics + '_Atg.npy', case.Atg)
    ex.add_artifact(case.WV_path + specifics + '_Atg.npy')
    np.save(case.WV_path + specifics + '_DDC_norm.npy', case.DDC_norm)
    ex.add_artifact(case.WV_path + specifics + '_DDC_norm.npy')


    print('Finished AF-FDK')




    case.table()
    latex_table = open(case.WV_path + specifics + '_latex_table.txt', 'w')
    latex_table.write(case.table_latex)
    latex_table.close()
    ex.add_artifact(case.WV_path + specifics + '_latex_table.txt')

    np.save(case.WV_path + specifics + '_seg_err.npy', seg_err)
    ex.add_artifact(case.WV_path + specifics + '_seg_err.npy')
    np.save(case.WV_path + specifics + '_seg_err_list.npy', seg_err_list)
    ex.add_artifact(case.WV_path + specifics + '_seg_err_list.npy')

    np.save(case.WV_path + specifics + '_pore_dist.npy', pore_dist)
    ex.add_artifact(case.WV_path + specifics + '_pore_dist.npy')
    case = None
    gc.collect()
    return pore_dist