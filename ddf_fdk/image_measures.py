#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:51:06 2018

@author: lagerwer
"""


import numpy as np
from scipy.ndimage import uniform_filter
import gc
from skimage.filters import threshold_otsu
import scipy.ndimage.morphology as sp
import scipy as sps
from scipy import ndimage
import pylab


# %%
def porosity(im):
    im = np.array(im, dtype=int)
    Vp = np.sum(im == 1)
    Vs = np.sum(im == 0)
    e = Vp/(Vs + Vp)
    return e

def comp_rSegErr(S, S_GT):
    err = np.linalg.norm(np.ravel(S) * 1 - np.ravel(S_GT) * 1, 1) / np.size(S)
    return err

def comp_outer_reg(X):
    vox = np.size(X, 0)
    out_reg = sp.binary_dilation(np.asarray(X), iterations=vox // 16)
    return sp.binary_erosion(np.asarray(out_reg), iterations=vox // 16)
    
def comp_porosity(X):
    fig = comp_outer_reg(np.asarray(X) * 1)
    fig = np.invert(fig) * 2 + np.invert(np.asarray(X) * 1)
    return porosity(fig)
    
def pore_size_distr(seg, mask, bin_size, number_bins=10, show_particles=False):
    part_count = np.zeros((number_bins))
    mid = np.size(seg, 0) // 2
    label_array = np.zeros((np.shape(seg)))
    a = np.invert(np.asarray(seg)) * mask
    for i in range(number_bins):
        label_array, part_count[i] = ndimage.measurements.label(a)
        
        a = sp.binary_erosion(np.asarray(a), iterations=bin_size)
        if show_particles:
            pylab.figure()
            pylab.imshow(label_array[:, :, mid])        
        if part_count[i] <= 1:
            print('No larger objects anymore')
            break
        
    return part_count

def part_count_to_distr(part_count, bin_size, number_bins):
    pore_dist = np.zeros(number_bins)
    for i in range(number_bins - 1):
        if part_count[i] >= part_count[i + 1]:
            pore_dist[i] = part_count[i] - part_count[i + 1]
        else:
            pore_dist[i] = part_count[i]
            Err = True
            break
        Err = False
    if Err:
        return pore_dist
    else:
        pore_dist[-1] = part_count[-1]
    return pore_dist
# %%    
# ! ! ! TODO: Add gaussian weights for SSIM
def comp_ssim(X, Y, C1, C2, filt_size):
        # X, Y
        ux = uniform_filter(X, size=filt_size)
        uxx = uniform_filter(X * X, size=filt_size)
        vx = ux.copy()
        # X, Y, ux, uxx, vx
        vx *= ux
        vx -= uxx
        vx *= -1
        del uxx
        # X, Y, ux, vx

        uy = uniform_filter(Y, size=filt_size)
        uyy = uniform_filter(Y * Y, size=filt_size)
        uxy = uniform_filter(X * Y, size=filt_size)
        # X, Y, ux, vx, uy, uyy, uxy
        del X, Y
        vy = uy.copy()
        # ux, vx, vy, uy, uyy, uxy
        vy *= uy
        vy -= uyy
        vy *= -1
        del uyy
        # ux, vx, vy, uy, uxy

        vxy = ux.copy()
        # ux, vx, vy, uy, uxy, vxy
        vxy *= uy
        vxy -= uxy
        vxy *= -1
        del uxy
        # ux, vx, vy, uy, vxy

        # Ik denk dat ik dfe eerste twee om kan draaien
        # first term numerator
        A = ux.copy()   # A = ux
        A *= 2 * uy     # A = 2 * ux * uy
        A += C1         # A = 2 * ux * uy + C1
        # second term numerator
        vxy *= 2        # vxy = 2*vxy
        vxy += C2       # vxy = 2*vxy + C2
        # numerator
        A *= vxy        # A = (2 * ux * uy + C1)*(2*vxy + C2)
        del vxy
        # first term denumerator
        ux *= ux        # ux = ux ** 2
        uy *= uy        # uy = uy ** 2
        ux += uy        # ux = ux ** 2 + uy ** 2
        del uy
        ux += C1        # ux = ux ** 2 + uy ** 2 + C1
        # second term denumerator
        vx += vy        # vx = vx + vy
        del vy
        vx += C2        # vx = vx + vy + C2
        ux *= vx        # ux = (ux ** 2 + uy ** 2 + C1)*(vx + vy + C2)
        del vx
        # Create the pointwise SSIM
        A /= ux
        # A = (2 * ux * uy + C1)*(2*vxy + C2)/ ((ux ** 2 + uy ** 2 + C1)*(vx + vy + C2))
        del ux
        return A

def bd_dim_split(case, quarter, mid_filt):
    if case == 0:
        bd_min = None
        bd_max = quarter + mid_filt
    elif case == 1:
        bd_min = quarter - mid_filt
        bd_max = 2 * quarter + mid_filt
    elif case == 2:
        bd_min = 2 * quarter - mid_filt
        bd_max = 3 * quarter + mid_filt
    elif case == 3:
        bd_min = 3 * quarter - mid_filt
        bd_max = None
    return bd_min, bd_max

def bd_dim_pblock(case, quarter, mid_filt):
    if case == 0:
        bd_min = None
        bd_max = quarter
    elif case == 1 or case == 2:
        bd_min = mid_filt
        bd_max = -mid_filt
    elif case == 3:
        bd_min = mid_filt
        bd_max = None
    return bd_min, bd_max

def bd_dim_pobj(case, quarter, mid_filt):
    if case == 0:
        bd_min = None
        bd_max = quarter
    elif case == 1:
        bd_min = quarter
        bd_max = 2 * quarter
    elif case == 2:
        bd_min = 2 * quarter
        bd_max = 3 * quarter
    elif case == 3:
        bd_min = 3 * quarter
        bd_max = None
    return bd_min, bd_max


def split_data(x, filt_size, cube):
    quarter = np.shape(x)[0] // 4
    mid_filt = (filt_size + 1) // 2
    min_x, max_x = bd_dim_split(cube[0], quarter, mid_filt)
    min_y, max_y = bd_dim_split(cube[1], quarter, mid_filt)
    min_z, max_z = bd_dim_split(cube[2], quarter, mid_filt)
    x_split = x[min_x: max_x, min_y: max_y, min_z: max_z]
    return x_split

def patch_data(x, x_block, filt_size, cube):
    quarter = np.shape(x)[0] // 4
    mid_filt = (filt_size + 1) // 2
    min_bx, max_bx = bd_dim_pblock(cube[0], quarter, mid_filt)
    min_by, max_by = bd_dim_pblock(cube[1], quarter, mid_filt)
    min_bz, max_bz = bd_dim_pblock(cube[2], quarter, mid_filt)
    min_ox, max_ox = bd_dim_pobj(cube[0], quarter, mid_filt)
    min_oy, max_oy = bd_dim_pobj(cube[1], quarter, mid_filt)
    min_oz, max_oz = bd_dim_pobj(cube[2], quarter, mid_filt)
    x[min_ox:max_ox, min_oy:max_oy, min_oz:max_oz] = \
            x_block[min_bx:max_bx, min_by:max_by, min_bz:max_bz]

def patch_ssim(X, Y, C1, C2, filt_size):
    ssim = np.zeros(np.shape(X))
    for i in range(4):
            for j in range(4):
                for k in range(4):
                    x_block = split_data(X, filt_size, [i, j, k])
                    y_block = split_data(Y, filt_size, [i, j, k])
                    x_block = comp_ssim(x_block, y_block, C1, C2, filt_size)
                    patch_data(ssim, x_block, filt_size, [i,j,k])
                    del x_block, y_block
    return ssim


# %%
def SSIM(x, PH, patch, filt_size, mask='yes'):
    dmin, dmax = ((np.min(PH.f), np.max(PH.f)))
    data_range = dmax - dmin

    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    X = np.asarray(x).astype(np.float32)
    x = None
    Y = np.asarray(PH.f).astype(np.float32)
    gc.collect()
    if patch == 0:
        ssim = comp_ssim(X, Y, C1, C2, filt_size)
    elif patch == 1:
        ssim = patch_ssim(X, Y, C1, C2, filt_size)
    X, Y = None, None
    gc.collect()
    if mask == 'yes':
        ssim = PH.mask(ssim)
        return np.sum(ssim) / PH.mask_size
    elif mask == 'no':
        return ssim.mean()


# %%
def MAR(fp_recon, data):
    fp_rec = np.asarray(fp_recon).flatten()
    dat = np.asarray(data).flatten()
    l1_dat = np.linalg.norm(dat, 1)
    return np.linalg.norm(fp_rec - dat, 1)/ (l1_dat)


def MSR(fp_recon, data):
    fp_rec = np.asarray(fp_recon).flatten()
    dat = np.asarray(data).flatten()
    l2_dat = np.linalg.norm(dat, 2)
    return  np.linalg.norm(fp_rec - dat, 2) ** 2 / (l2_dat**2)


def MAE(recon, PH, mask='yes'):
    if mask == 'yes':
        rec = np.asarray(PH.mask(recon)).flatten()
    else:
        rec = np.asarray(recon).flatten()
    gt = np.asarray(PH.f).flatten()
    l1_gt = np.linalg.norm(gt, 1)
    return np.linalg.norm(rec - gt, 1)/(l1_gt)


def MSE(recon, PH, mask='yes'):
    if mask == 'yes':
        rec = np.asarray(PH.mask(recon)).flatten()
    else:
        rec = np.asarray(recon).flatten()
    gt = np.asarray(PH.f).flatten()
    l2_gt = np.linalg.norm(gt, 2)
    return np.linalg.norm(rec - gt, 2) ** 2 / (l2_gt**2)

