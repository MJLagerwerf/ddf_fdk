#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:55:46 2019

@author: lagerwer
"""

import numpy as np
import scipy.ndimage as sp
import scipy.interpolate as si

import gc
import time
import os
import re
import astra

from . import read_experimental_data as red

# %%
def load_and_preprocess_real_data(path, dset, sc, redo=False, zoom=False):
    preprocess_data(path, dset, sc, redo)
    proc_path = path + 'processed_data/'
    if sc == 1:
        if not (os.path.exists(proc_path + 'ground_truth.npy') and \
            os.path.exists(proc_path + 'mask.npy')) or redo:
            print('Computing mask and ground truth for this dataset')
            make_golden_standard_and_mask(path, sc, zoom)
        else:
            print('Already computed mask and ground truth for this dataset')

        dataset = {'g' : proc_path + 'g_'+ dset + '.npy',
                   'ground_truth' : proc_path + 'ground_truth.npy',
                   'mask' : proc_path + 'mask.npy'}   
    else:
        if not (os.path.exists(proc_path + 'ground_truth_sc' + str(sc) + \
                               '.npy') and os.path.exists(proc_path + \
                                     'mask_sc' +str(sc) +'.npy')) or redo:
            print('Computing mask and ground truth for this dataset')
            make_golden_standard_and_mask(path, sc, zoom)
        else:
            print('Already computed mask and ground truth for this dataset')
            
        dataset = {'g' : proc_path + 'g_' + dset + '_sc' + str(sc) + '.npy',
                   'ground_truth' : proc_path + 'ground_truth_sc' + str(sc) + \
                   '.npy',
                   'mask' : proc_path + 'mask_sc' + str(sc) + '.npy'}
    return dataset

# %%
def load_meta(load_path, sc):
    meta = {}
    f = open(load_path + 'data settings XRE.txt',  'r')
    meta_raw = f.read()
    labels = ['Pixel size', 'SOD', 'SDD']
    keys = ['pix_size', 's2o', 's2d']
    
    for i in range(len(labels)):
        for m in re.findall(labels[i] + ' = \"([0-9\.]*)\"', meta_raw):
            if keys[i] == 'pix_size':
                meta[keys[i]] = float(m) * 0.1 * sc
            else:
                meta[keys[i]] = float(m) * 0.1
                
    meta['o2d'] = meta['s2d'] - meta['s2o']
    return meta


# %%
def find_center(P_a, meta):
    pix_size = meta['pix_size']
    s2d = meta['s2d']
    s2o = meta['s2o']
    N_a, N_u = np.shape(P_a)
    angles = np.linspace(0, 2 * np.pi, N_a) 
    det_space_u = np.linspace(- N_u * pix_size / 2, N_u * pix_size / 2, N_u)


    # Start settings:
    midpoint = 0
    precision = 10 ** (-np.arange(5, dtype='float') - 1)
    # %%
    
    for i in range(len(precision)):
        COR = np.arange(midpoint - 10 * precision[i], midpoint + 10 * precision[i],
                        precision[i])
        M = np.zeros(len(COR))
        print(midpoint)
        for j in range(len(COR)):
            gamma = np.arctan(det_space_u / s2d)
            gamma_c = np.arctan(COR[j] / s2d)
            gamma_i = gamma - gamma_c
            beta = np.pi - 2 * gamma_i
            
            s2 = s2d * np.tan(2 * gamma_c - gamma)
    
            s2t = np.tile(s2, (N_a, 1))
            ang = (np.tile(angles, (N_u, 1)).T + np.tile(beta, (N_a, 1))) \
                        % (2 * np.pi)
            IO = si.RegularGridInterpolator((angles, det_space_u), P_a, 
                                            bounds_error=False, fill_value=None)
            P_ab = np.reshape(IO(np.concatenate([[ang.ravel(),], [s2t.ravel(),]],
                                                  axis=0).T), (N_a, N_u))
            
            M[j] = np.linalg.norm(P_ab - P_a) ** 2
            
        midpoint = COR[np.argmin(M)]
    
    return midpoint * -s2o / s2d / pix_size


# %%
def preprocess_data(path, dset, sc, redo):
    start = time.time()
    
    proc_path = path + 'processed_data/'
    if not os.path.exists(proc_path):
        os.makedirs(proc_path)
    # Sampling scheme to reduce the size of the data
    # Save the vector
    if sc == 1:
        save_path = proc_path + 'g_' + dset
    else:
        save_path = proc_path + 'g_' + dset + '_sc' + str(sc)
    if not os.path.exists(save_path + '.npy') or redo:
        meta = load_meta(path + dset + '/', sc)
        sampling = [sc, sc]
        dark = red.read_raw(path + dset, 'di0', sample=sampling)
        flat = red.read_raw(path + dset, 'io0', sample=sampling)
        proj = red.read_raw(path + dset, 'scan_0', sample=sampling)
        # if there is a dead pixel, give it the minimum photon count from proj
        max_photon_count = proj.max()
        proj[proj == 0] = max_photon_count + 1
        min_photon_count = proj.min()
        proj[proj == max_photon_count + 1] = min_photon_count
        # %%
        proj = (proj - dark) / (flat.mean(0) - dark)
        proj = -np.log(proj)
        
        # We know that the first and the last projection angle overlap,
        # so only consider the first
        proj = proj[:-1, :, :]
        
        # Make sure that the amount of detectors rows and collums are even
        if np.size(proj, 2) % 2 is not 0:
            line_u = np.zeros((proj.shape[0], proj.shape[1], 1))
            proj = np.concatenate((proj, line_u), 2)
        if np.size(proj, 1) % 2 is not 0:
            line_u = np.zeros((proj.shape[0], 1, proj.shape[2]))
            proj = np.concatenate((proj, line_u), 1)
        
        # Compute the center of rotation difference in pixels
#        d_COR = center_of_mass(proj ** 2)[2] - proj.shape[2]/2 ! ! ! Outdated
        mid_v = proj.shape[1] // 2
        center_slice = proj[:, mid_v, :]
        
        d_COR = find_center(center_slice, meta)
        print('Center of rotation wrt the current center', d_COR)
        if np.abs(d_COR) > 0.5:
            proj = sp.shift(proj, [0, 0, d_COR], mode='nearest')
            center_slice = proj[:, mid_v, :]
            d_COR = find_center(center_slice, meta)
            print('Center of rotation wrt the current center', d_COR)
        else:
            print('Shift is smaller than half a pixel')
            pass
        # Transpose of the detector to fit the input format and turn it 180 degrees
        proj = np.transpose(proj, (0, 2, 1))
        proj = proj[:,:, ::-1]

        # Save the vector
        np.save(save_path, proj)
    
        print('Finished preprocessing and saving', time.time() - start)
    else:
        print('Already have a preprocessed g for this case')


def make_golden_standard_and_mask(path, sc=1, zoom=False):
    # %%
    start = time.time()
    preprocess_data(path, 'good', sc, redo=False)
    proc_path = path + 'processed_data/'
    
    if sc == 1:
        g = np.load(proc_path + 'g_good.npy')
    else:
        g = np.load(proc_path + 'g_good_sc' + str(sc) + '.npy')
    g = np.transpose(g, (2, 0, 1))
    gc.collect()
    # %%
    meta = load_meta(path + 'good/', sc)
    vox = g.shape[0]
    dpixsize = meta['pix_size']
    s2d = meta['s2d']
    s2o = meta['s2o']
    o2d = meta['o2d']
    if zoom:
        magn = 1
    else:
        magn = s2o / s2d
    vox_size = dpixsize * magn
    minsize = -vox * vox_size / 2 
    maxsize = vox * vox_size / 2 
    vol_geom = astra.create_vol_geom(vox, vox, vox, minsize, maxsize,
                                     minsize, maxsize, minsize, maxsize)
    
    ang = np.shape(g)[1]
    angles = np.linspace(np.pi/ang, (2 + 1 / ang) * np.pi, ang, False)
    proj_geom = astra.create_proj_geom('cone', dpixsize, dpixsize,
                                       np.shape(g)[0], np.shape(g)[2],
                                       angles, s2o, o2d)
    
    # %%
    # Create projection data from this
    #proj_id, proj_data = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)
    proj_id = astra.data3d.create('-proj3d', proj_geom, g)
    #proj_id = astra.data3d.create('-proj3d', proj_geom, vol_geom)
    projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    # %%
    
    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)
    
    # Set up the parameters for a reconstruction algorithm using the GPU
    #W = astra.OpTomo(proj_id)
    astra.plugin.register(astra.plugins.SIRTPlugin)
    cfg = astra.astra_dict('SIRT-PLUGIN')
#    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['ProjectorId'] = projector_id
    cfg['option']= {}
    cfg['option']['MinConstraint'] = 0
    
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    
    # SIRT
    astra.algorithm.run(alg_id, 300)
    
    # Get the result
    rec = astra.data3d.get(rec_id)
    rec = np.transpose(rec, (2, 1, 0))
    save = np.zeros((3, vox, vox))
    save[0, :, :], save[1, :, :] = rec[:, :, vox // 2], rec[:, vox // 2, :]
    save[2, :, :] = rec[vox // 2, :, :]
    np.save(proc_path + 'rec_ax_SIRT300', save)
    # %%
    if sc == 1:
        np.save(proc_path + 'rec_SIRT300', rec)
    else:
        np.save(proc_path + 'rec_SIRT300_sc' + str(sc), rec)
    end = time.time()
    print((end-start), 'Finished SIRT 300 reconstructionn')
    ## Clean up. Note that GPU memory is tied up in the algorithm object,
    ## and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
    astra.projector3d.delete(projector_id)
    
    # %%
    rec *= (rec > 0.03)
    edge = vox // 32
    edge_t = np.zeros(np.shape(rec))
    edge_t[edge:-edge, edge:-edge, edge:-edge] = 1
    rec *= edge_t
    del edge_t
    save[0, :, :], save[1, :, :] = rec[:, :, vox // 2], rec[:, vox // 2, :]
    save[2, :, :] = rec[vox // 2, :, :]
    np.save(proc_path + 'GT_ax', save)
    diter = int(1.5 * 2 **(np.log2(vox) - 5))
    it = 5
    mask = sp.binary_erosion(rec, iterations=it)
    mask = sp.binary_dilation(mask, iterations=diter + it)
    save[0, :, :], save[1, :, :] = mask[:, :, vox // 2], mask[:, vox // 2, :]
    save[2, :, :] = mask[vox // 2, :, :]
    np.save(proc_path + 'mask_ax', save)
    
    # %%
    if sc == 1:
        np.save(proc_path + 'ground_truth.npy', rec)
        np.save(proc_path + 'mask.npy', mask)
        
    else:
        np.save(proc_path + 'ground_truth_sc' + str(sc) + '.npy' , rec)
        np.save(proc_path + 'mask_sc' + str(sc) + '.npy', mask)
    t3 = time.time()
    print(t3-end, 'Finished computing mask and ground truth')
