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

def get_axis(rec):
    mid = np.shape(rec)[0] // 2
    xy = rec[mid, :, :]
    xz = rec[:, mid, :]
    yz = rec[:, :, mid]
    return [[xy], [xz], [yz]]
    

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
    I0 = [2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14, 2 ** 16,
          2 ** 18, 2 ** 20]
    noise = ['Poisson', I0[it_i]]

    # Source radius
    src_rad = 10

    lp = '/export/scratch2/lagerwer/NNFDK/phantoms/'
    f_load_path = None
    g_load_path = None

    num_bins = 10
    bin_size = 5

    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    specifics = 'I0' + str(I0[it_i])
    

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
    
@ex.capture
def log_gen_variables(results, arr):
    arr = np.append(arr, [results,], axis=0)
    return arr
    

# %%
@ex.automain
def main(specifics, bin_size, num_bins):
    if not os.path.exists('AFFDK_results'):
        os.makedirs('AFFDK_results')
    t2 = time.time()
    # Create a data object
    case = CT()
    t3 = time.time()
    print(t3 - t2, 'seconds to initialize CT object')

    part_count_list = np.zeros((0, num_bins))
    pore_dist_list = np.zeros((0, num_bins))
    save_and_add_artifact(f'{case.WV_path}_g.npy', case.g)

    f = 'Shepp-Logan'
    LP_filts = [['Gauss', 8], ['Gauss', 5], ['Bin', 2], ['Bin', 5]]
    
    rec = case.FDK.do(f, compute_results=False)
    seg, part_count, pore_dist = ddf.do_seg_and_pore_dist(rec, bin_size,
                                                          num_bins)
    print(np.shape(part_count))
    print(np.shape(pore_dist))
    save_and_add_artifact(f'{case.WV_path}{specifics}_FDKSL_seg_full.npy', seg)
    save_and_add_artifact(f'{case.WV_path}{specifics}_FDKSL_seg.npy',
                          get_axis(seg))
    log_gen_variables(part_count, part_count_list)
    log_gen_variables(pore_dist, pore_dist_list)

    for lp in LP_filts:
        rec = case.FDK.filt_LP(f, lp, compute_results=False)
        seg, part_count, pore_dist = ddf.do_seg_and_pore_dist(rec, bin_size,
                                                          num_bins)
        log_gen_variables(part_count, part_count_list)
        log_gen_variables(pore_dist, pore_dist_list)

        if lp[0] == 'Gauss':
            save_and_add_artifact(f'{case.WV_path}{specifics}'+ \
                                  f'_FDKSL_GS{lp[1]}_seg.npy',
                                  get_axis(seg))
            save_and_add_artifact(f'{case.WV_path}{specifics}'+ \
                                  f'_FDKSL_GS{lp[1]}_seg_full.npy',
                                  seg)
        elif lp[0] == 'Bin':
            save_and_add_artifact(f'{case.WV_path}{specifics}'+ \
                                  f'_FDKSL_BN{lp[1]}_seg.npy',
                                  get_axis(seg))
            save_and_add_artifact(f'{case.WV_path}{specifics}'+ \
                                  f'_FDKSL_BN{lp[1]}_seg_full.npy',
                                  seg)

    print('Finished FDKs')
    
    
    rec = case.TFDK.do(lam='optim', compute_results=False)
    seg, part_count, pore_dist = ddf.do_seg_and_pore_dist(rec, bin_size,
                                                  num_bins)
    save_and_add_artifact(f'{case.WV_path}{specifics}_TFDK_seg.npy',
                          get_axis(seg))
    save_and_add_artifact(f'{case.WV_path}{specifics}_TFDK_seg_full.npy',
                          seg)

    log_gen_variables(part_count, part_count_list)
    log_gen_variables(pore_dist, pore_dist_list)
    save_and_add_artifact(f'{case.WV_path}{specifics}_part_count.npy',
                          part_count_list)
    save_and_add_artifact(f'{case.WV_path}{specifics}_pore_dist.npy',
                          pore_dist_list)
    save_and_add_artifact(f'{case.WV_path}{specifics}_AtA.npy', case.AtA)
    save_and_add_artifact(f'{case.WV_path}{specifics}_Atg.npy', case.Atg)
    save_and_add_artifact(f'{case.WV_path}{specifics}_DDC_norm.npy',
                          case.DDC_norm)

    print('Finished MR-FDK')

    save_table(case, f'{case.WV_path}{specifics}')

    case = None
    gc.collect()
    return pore_dist