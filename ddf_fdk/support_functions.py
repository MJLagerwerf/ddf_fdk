#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:03:26 2018

@author: lagerwer
"""


import odl
import numpy as np
import pylab
import pyfftw
import time
import os
import shutil
from tempfile import mkstemp
from scipy.ndimage import gaussian_filter

from .phantom_class import phantom
from . import CCB_CT_class as CT
# %% Working vars function
class working_var_map:
    def __init__(self):
        base_WV_path = 'python_data/working_vars'
        _, temp_name = mkstemp()
        self.WV_path = base_WV_path + temp_name + '/'
        os.makedirs(self.WV_path)


    def __del__(self):
        shutil.rmtree(self.WV_path, ignore_errors=True)
# %%
class SymOp(odl.Operator):
    def __init__(self, dom, ran):
        odl.Operator.__init__(self, domain=dom, range=ran, linear=True)
        self.dom = dom
        self.ran = ran

    def _call(self, x):
        u = np.asarray(x)
        return np.concatenate([u[-1:], u[::-1], u[1:]])

    @property
    def adjoint(self):
        functional = self

        class AdjSymOp(odl.Operator):
            def __init__(self):
                super().__init__(domain=functional.range,
                                 range=functional.domain, linear=True)

            def _call(self, x):
                u = np.asarray(x)
                mid = int(np.floor(np.size(u) / 2))
                x_out = u[mid:] + u[mid-1::-1]
                return x_out

            @property
            def adjoint(self):
                return functional
        return AdjSymOp()


# %%
isft = np.fft.ifftshift
class ConvolutionOp(odl.Operator):
    def __init__(self, filter_space, data_space, det_space, w_detv, g_w, a_wc):
        odl.Operator.__init__(self, domain=filter_space, range=data_space, linear=True)

        self.rs_detu = int(2 ** (np.ceil(np.log2(filter_space.size)) + 1))
        # set =filter.size if filter is twice the detector
        self.frs_detu = self.rs_detu // 2 + 1
        self.rs_filt = odl.ResizingOperator(filter_space,
                                            ran_shp=(self.rs_detu,))
        self.gw = np.asarray(g_w)
        self.a_wc = a_wc
        self.w_detv = w_detv
#        self.rs_filt.domain.weighting.const
        self.det_tr_space = odl.uniform_discr([det_space.min_pt[1],
                                               det_space.min_pt[0]],
                                                [det_space.max_pt[1],
                                                 det_space.max_pt[0]],
                                                 (np.size(det_space, 1),
                                                  np.size(det_space, 0)))
        self.rs_det = odl.ResizingOperator(self.det_tr_space,
                                    ran_shp=(np.size(self.range, 2),
                                             self.rs_detu))
        self.flt = 'float32'
        self.clx = 'complex64'
        self.flt1 = 'float64'
        self.clx1 = 'complex128'
        thrds = 8
        self.d_a = pyfftw.empty_aligned((np.size(self.range, 2), self.rs_detu),
                                        dtype=self.flt)
        self.d_b = pyfftw.empty_aligned((np.size(self.range, 2), self.frs_detu),
                                        dtype=self.clx)
        self.fft_d = pyfftw.FFTW(self.d_a, self.d_b, axes=(1,), threads=thrds)
        self.id_a = pyfftw.empty_aligned((np.size(self.range, 2), self.frs_detu),
                                        dtype=self.clx)
        self.id_b = pyfftw.empty_aligned((np.size(self.range, 2), self.rs_detu),
                                        dtype=self.flt)
        self.ifft_d = pyfftw.FFTW(self.id_a, self.id_b, axes=(1,),
                                  threads=thrds, direction='FFTW_BACKWARD')


        self.f_a = pyfftw.empty_aligned(self.rs_detu, dtype=self.flt)
        self.f_b = pyfftw.empty_aligned(self.frs_detu, dtype=self.clx)
        self.fft_f = pyfftw.FFTW(self.f_a, self.f_b)
        self.if_a = pyfftw.empty_aligned(self.frs_detu, dtype=self.clx1)
        self.if_b = pyfftw.empty_aligned(self.rs_detu, dtype=self.flt1)
        self.ifft_f = pyfftw.FFTW(self.if_a, self.if_b,
                                  direction='FFTW_BACKWARD')

    def _call(self, x, out):
        # Do the convolution
        # overwrite f_b, fourier transform filter
        y = np.asarray(out)
        self.fft_f(isft(self.rs_filt(x)))
        # Create a detector sized stack of fourier tranformed filters
        hf_stack = np.zeros((np.size(self.range, 2), self.frs_detu),
                            dtype=self.clx)
        # Resize the weighted data
        tmp1 = self.rs_det.range.element()
        # Stack the filter
        for j in range(np.size(self.range, 2)):
            hf_stack[j, :] = self.f_b
        for i in range(np.size(self.range, 0)):
            # Fourier transform of the data with angle i
            self.rs_det(self.gw[i,:,:].T, out=tmp1)
            self.fft_d(tmp1)        # overwrite d_b
            # Product of the filter and the data in fspace
            self.d_b *= hf_stack
            # overwrite id_b, inverse fourier transform product
            self.ifft_d(self.d_b)
            # Overwrite the data with the convolution
            y[i, :, :] = np.asarray(self.rs_det.inverse(self.id_b)).T
        out[:] = y

    @property
    def adjoint(self):
        f = self

        class AdjStackingOp(odl.Operator):
            def __init__(self):
                super().__init__(domain=f.range, range=f.domain, linear=True)

            def _call(self, x, out):
                # Create a vector for the convolution resul
                tmp3 = np.zeros(f.rs_detu, dtype=f.flt1)
                # Resize the input
                x = np.asarray(x)
                tmp = f.rs_det.range.element()
                tmp2 = f.rs_det.range.element()
                for i in range(np.size(f.range, 0)):
                    # fourier transform the data for angle i
                    (f.rs_det(f.gw[i, :, :].T, out = tmp))
                    f.fft_d(tmp)       # overwrite d_b
                    tmp1 = np.conj(f.d_b)
                    # fourier transform the input for angle i
                    (f.rs_det(x[i, :, :].T, out=tmp2))
                    f.fft_d(tmp2)         # overwrite d_b
                    # compute product
                    tmp1 *= f.d_b
                    # inverse fourier transform product
                    tmp4 = np.sum(tmp1, 0)
                    f.ifft_f(tmp4)             # Overwrite if_b
                    tmp3 += f.if_b
                tmp1, tmp2, tmp4 = None, None, None
                tmp3 *= f.w_detv * f.a_wc
                tmp4 = isft(tmp3)
                f.rs_filt.inverse(tmp4, out=out)
            @property
            def adjoint(self):
                return f
        return AdjStackingOp()

    def conv_hf(self, hf):
        y = np.zeros(np.shape(self.range))
        hf_stack = np.zeros((np.size(self.range, 2), self.frs_detu),
                            dtype=self.clx)
        # Resize the weighted data
        tmp1 = self.rs_det.range.element()
        # Stack the filter
        for j in range(np.size(self.range, 2)):
            hf_stack[j, :] = hf
        for i in range(np.size(self.range, 0)):
            # Fourier transform of the data with angle i
            self.rs_det(self.gw[i,:,:].T, out=tmp1)
            self.fft_d(tmp1)        # overwrite d_b
            # Product of the filter and the data in fspace
            self.d_b *= hf_stack
            # overwrite id_b, inverse fourier transform product
            self.ifft_d(self.d_b)
            # Overwrite the data with the convolution
            y[i, :, :] = np.asarray(self.rs_det.inverse(self.id_b)).T
        return self.range.element(y) # Invert the resizing


# %% Exponential binning funciton
def ExpBin(bin_param, size_filter_space):
    i = 0
    d = []
    B = [0]
    width = 1
    while i < int(size_filter_space/2):
        if i < bin_param:
            d += [width]
            i += 1
            B += [np.sum(d)]
        else:
            d += [width]
            width *= 2
            i += width
            B += [np.sum(d)]
    B += [int(size_filter_space/2)]
    return np.asarray(B)


# %%
def ExpOp_builder(bin_param, filter_space, interp):
    # Create binning scheme
    if interp == 'Full':
        spf_space = filter_space
        Exp_op = odl.IdentityOperator(filter_space)
    elif interp == 'uniform':
        # Create binning scheme
        dpix = np.size(filter_space)
        dsize = filter_space.max_pt
        filt_bin_space = odl.uniform_discr(-dsize, dsize, dpix // (bin_param))
        spf_space = odl.uniform_discr(0, dsize, dpix //(2 * bin_param))
        resamp = odl.Resampling(filt_bin_space, filter_space)
        sym = SymOp(spf_space, filt_bin_space)
        Exp_op = resamp * sym
    else:
        if interp == 'constant':
            interp = 'nearest'
        elif interp == 'linear':
            pass
        else:
            raise ValueError('unknown `expansion operator type` ({})'
                         ''.format(interp))
        B = ExpBin(bin_param, np.size(filter_space)) * \
                        filter_space.weighting.const
        B[-1] -= 1 / 2 * filter_space.weighting.const

        # Create sparse filter space
        spf_part = odl.nonuniform_partition(B, min_pt=0, max_pt=B[-1])
        spf_weight = np.ravel(np.multiply.reduce(np.meshgrid(
                *spf_part.cell_sizes_vecs)))
        spf_fspace = odl.FunctionSpace(spf_part.set)
        spf_space = odl.DiscreteLp(spf_fspace, spf_part, odl.rn(spf_part.size,
                                        weighting=spf_weight), interp=interp)
        filt_pos_part = odl.uniform_partition(0, B[-1],
                                              int(np.size(filter_space) / 2))

        filt_pos_space = odl.uniform_discr_frompartition(filt_pos_part,
                                                         dtype='float64')
        lin_interp = odl.Resampling(spf_space, filt_pos_space)

        # Create symmetry operator
        sym = SymOp(filt_pos_space, filter_space)

        # Create sparse filter operator
        Exp_op = sym * lin_interp
    return spf_space, Exp_op


# %% Function to compute the FDK weighting
def FDK_weighting(detecpix, det_space, w_detu, w_detv, src_rad, det_rad=0):
    midu = int(detecpix[0]/2)
    midv = int(detecpix[1]/2)
    rho = src_rad + det_rad
    w_FDK = np.ones((np.shape(det_space))) * rho ** 2
    for i in range(np.size(w_FDK, 0)):
        w_FDK[i, :] += ((-midu + i + 1/2) * w_detu) ** 2

    for j in range(np.size(w_FDK, 1)):
        w_FDK[:, j] += ((-midv + j + 1/2) * w_detv) ** 2

    return rho * np.sqrt(1 / w_FDK)


# %% Ramp filter
def ramp_filt(rs_detu):
    mid_det = int(rs_detu / 2)
    filt_impr = np.zeros(mid_det + 1)
    n = np.arange(mid_det + 1)
    tau = 1
    filt_impr[0] = 1 / (4* tau **2)
    filt_impr[1::2] = -1 / (np.pi ** 2 * n[1::2] ** 2* tau **2)
    filt_impr = np.append(filt_impr[:-1], np.flip(filt_impr[1:], 0))
    return filt_impr


# %%
def make_bin_LP(level):
    LP_base = np.array([1/2, 1/2])
    LP = LP_base.copy()
    if level == 1:
        return LP
    else:
        for i in range(level-1):
            LP = np.convolve(LP, LP_base)
        return LP
    
# %%
def low_pass_filter(h, LP_filt):
    h_LP = np.zeros(np.shape(h))
    if LP_filt[0] == 'Gauss':
        gaussian_filter(h, LP_filt[1], output=h_LP)
    elif LP_filt[0] == 'Bin':
        LP = make_bin_LP(LP_filt[1])
        h_LP = np.convolve(h, LP, mode='same')
    return h_LP


# %% Functions to go to a lower resolution
def integrate_data(g, factor=4):
    tot = np.zeros(np.shape(g[:, ::factor, ::factor]))
    for i1 in range(factor):
        for i2 in range(factor):
            tot += g[:, i1::factor, i2::factor]
    tot /= factor ** 2
    return tot

def subsamp_data(g, factor=4):
    return g[:, ::factor, ::factor]


# %%
def L1_distance(rec, ref):
    return np.linalg.norm(np.ravel(rec) - np.ravel(ref), 1)


# %%
def load_results(path, nMeth, nExp, files, spec, spec_var):
    Q = np.zeros((nMeth, nExp, 3))
    i = 0
    for f in files:
        Q[:, i, :] = np.load(path + str(f)+ '/' + spec + str(spec_var[i]) + '_Q.npy')
        i += 1
    return Q

# %%
def MTF_x(filts, data_obj):
    DO = phantom(data_obj.voxels, 'Plane_yz', data_obj.angles, None,
                     data_obj.src_rad, data_obj.det_rad)
    CTo = CT.CCB_CT(DO)
    CTo.init_algo()
    CTo.init_DDF_FDK()
    MTF_x_list = []
    mid = data_obj.voxels[0] // 2
    for f in filts:
        if type(f) == list:
            PlSF_x = CTo.FDK.filt_LP('Shepp-Logan', f, compute_results='no')
        elif type(f) == str:
            PlSF_x = CTo.FDK.do(f, compute_results='no')
        else:        
            PlSF_x = CTo.FDK_bin(f)
        MTF_x = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(PlSF_x), 
                                             axes=(0, 1, 2))))
        MTF_x_list += [MTF_x[mid:, mid, mid] / MTF_x[mid, mid, mid]]

    return MTF_x_list
    