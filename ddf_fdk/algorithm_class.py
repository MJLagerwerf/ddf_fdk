#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:17:12 2018

@author: lagerwer
"""
import odl
import numpy as np
import pylab
import tabulate
import time
import gc

from . import support_functions as sup
from . import image_measures as im
from .phantom_class import phantom
from .real_data_class import real_data
from . import CCB_CT_class as CT
from . import SIRT_ODL_astra_backend as SIRT_meth
from . import FDK_ODL_astra_backend as FDK_meth


# %%
def compute_QM(rec, CT_obj, measures):
    if hasattr(CT_obj.phantom, 'f'):
        Q = []
        # ! ! ! Not the relative MSE ! ! !
        if 'MSE' in measures:
            Q += [im.loss(rec, CT_obj.phantom, mask='yes')]
        else:
            Q += ['na']
        if 'MAE' in measures:
            Q += [im.MAE(rec, CT_obj.phantom, mask='yes')]
        else:
            Q += ['na']
        if 'SSIM' in measures:
            Q += [im.SSIM(rec, CT_obj.phantom, patch=1, filt_size=19,
                          mask='yes')]
        else:
            Q += ['na']
    else:
        Q = []
        Q += [im.MSR(CT_obj.FwP(rec), CT_obj.g)]
    return np.asarray(Q)[None, :]

class results:
    def __init__(self, algo, rec, measures, var, param, t_rec):
        self.rec_it = 0
        self.var = [var]
        self.param = [param]
        self.mid = np.shape(rec)[0] // 2
        self.rec_axis = [[rec[:, :, self.mid], rec[:, self.mid, :],
                     rec[self.mid, :, :]]]
        self.rec_time = [t_rec]
        self.Q = compute_QM(rec, algo.CT_obj, measures)
        self.Ql = []
        li = []
        li += [algo.method + ', ' + param]
        li += [i for i in self.Q[0, :]]
        self.Ql += [li]


    def add_results(self, algo, rec, measures, var, param, t_rec):
        self.rec_it += 1
        self.var += [var]
        self.param += [param]
        self.rec_axis += [[rec[:, :, self.mid], rec[:, self.mid, :],
                      rec[self.mid, :, :]]]
        self.rec_time += [t_rec]
        self.Q = np.append(self.Q,
                    compute_QM(rec, algo.CT_obj, measures), axis=0)
        li = []
        li += [algo.method + ', ' + param]
        li += [i for i in self.Q[self.rec_it, :]]
        self.Ql += [li]
# %%
class algorithm_class:
    def __init__(self, CT_obj, method='Name algorithm'):
        self.CT_obj = CT_obj
        self.method = method

    def comp_results(self, rec, measures, var, param, t_rec):
        if hasattr(self, 'results'):
            self.results.add_results(self, rec, measures, var, param, t_rec)
        else:
            self.results = results(self, rec, measures, var, param, t_rec)

    def show(self, rec_it=-1, clim=None, save_name=None, extension='.eps',
             fontsize=20):
        space = self.CT_obj.reco_space
        if clim == None:
            clim = [np.min(self.CT_obj.phantom.f),
                    np.max(self.CT_obj.phantom.f)]
        elif clim == False:
            clim = None
        xy, xz, yz = self.results.rec_axis[rec_it]
        fig, (ax1, ax2, ax3) = pylab.subplots(1, 3, figsize=[20, 6])
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ima = ax1.imshow(np.rot90(xy), clim=clim, extent=[space.min_pt[0],
                         space.max_pt[0],space.min_pt[1], space.max_pt[1]],
                            cmap='gray')
        ax1.set_xticks([],[])
        ax1.set_yticks([],[])
        ax1.set_xlabel('x', fontsize=fontsize)
        ax1.set_ylabel('y', fontsize=fontsize)
        ima = ax2.imshow(np.rot90(xz), clim=clim, extent=[space.min_pt[0],
                         space.max_pt[0],space.min_pt[2], space.max_pt[2]],
                            cmap='gray')
        ax2.set_xticks([],[])
        ax2.set_yticks([],[])
        ax2.set_xlabel('x', fontsize=fontsize)
        ax2.set_ylabel('z', fontsize=fontsize)
        ima = ax3.imshow(np.rot90(yz), clim=clim, extent=[space.min_pt[1],
                         space.max_pt[1],space.min_pt[2], space.max_pt[2]],
                            cmap='gray')
        ax3.set_xlabel('y', fontsize=fontsize)
        ax3.set_ylabel('z', fontsize=fontsize)
        ax3.set_xticks([],[])
        ax3.set_yticks([],[])
        fig.colorbar(ima, ax=(ax1, ax2, ax3))
        if save_name is None:
            fig.suptitle(self.method + ', ' + self.results.param[rec_it],
                         fontsize=fontsize+2)
        fig.show()
        if save_name is not None:
            pylab.savefig(save_name+extension, bbox_inches='tight')

    def table(self):
        headers = [' ', 'MSE', 'MAE_msk', 'SSIM_msk']
        print(tabulate.tabulate(self.results.Ql, headers,
                                tablefmt="fancy_grid"))
        self.table_latex = tabulate.tabulate(self.results.Ql, headers,
                                             tablefmt="latex")

    def plot_filt(self, h, fontsize=20):
        hf = np.real(np.asarray(self.CT_obj.pd_FFT(h)))
        xf = np.asarray(self.CT_obj.fourier_filter_space.grid)
        x = np.asarray(self.CT_obj.filter_space.grid)
        fig, (ax1, ax2) = pylab.subplots(1, 2, figsize=[15, 6])
        ax1.plot(x, h)
        ax1.set_title('Filter', fontsize=fontsize)
        ax1.set_ylabel('$h(u)$', fontsize=fontsize)
        ax1.set_xlabel('$u$', fontsize=fontsize)
        ax2.plot(xf, hf)
        ax2.set_title('Fourier transformed filter', fontsize=fontsize)
        ax2.set_ylabel('$\hat{h}(\omega)$', fontsize=fontsize)
        ax2.set_xlabel('$\omega$', fontsize=fontsize)
        fig.show()
# %%
class FDK_class(algorithm_class):
    def __init__(self, CT_obj):
        self.CT_obj = CT_obj
        self.method = 'FDK'

    def FDK_hf(self, hf):
        return self.CT_obj.BwP(self.CT_obj.conv_op.conv_hf(hf))


    def FDK_filt(self, filt_type):
        filt = np.real(np.fft.rfft(sup.ramp_filt(self.CT_obj.rs_detu)))
        freq = 2 * np.arange(len(filt))/(self.CT_obj.rs_detu)
        if filt_type == 'Ram-Lak':
            pass
        elif filt_type == 'Shepp-Logan':
            filt = filt * np.sinc(freq / 2)
        elif filt_type == 'Cosine':
            filt = filt * np.cos(freq * np.pi / 2)
        elif filt_type == 'Hamming':
            filt = filt * (0.54 + 0.46 * np.cos(freq * np.pi))
        elif filt_type == 'Hann':
            filt = filt * (np.cos(freq * np.pi / 2) ** 2)
        else:
            raise ValueError('unknown `filter_type` ({})'
                         ''.format(filt_type))
        weight = 1 / 2 / self.CT_obj.w_detu
        return (filt * weight)

    def do(self, filt_type, compute_results=True,
           measures=['MSE', 'MAE', 'SSIM'], astra=True):
        if hasattr(self.CT_obj, 'offset'):
            offset = self.CT_obj.offset
            if astra == False:
                raise ValueError('You need ASTRA to do a rotational offset')
        else:
            offset = 0
        t = time.time()
        hf = self.FDK_filt(filt_type)
        if astra:
            rec = self.CT_obj.reco_space.element(
                    FDK_meth.FDK_astra(self.CT_obj.g, hf,
                                       self.CT_obj.geometry,
                                       self.CT_obj.reco_space,
                                       self.CT_obj.w_detu, ang_offset=offset))
        else:
            rec = self.FDK_hf(hf)
        t_rec = time.time() - t
        if compute_results:
            self.comp_results(rec, measures, hf, filt_type, t_rec)
        else:
            return rec

    def filt_inp_do(self, h, filt_name, compute_results=True,
                    measures=['MSE', 'MAE', 'SSIM']):
        t = time.time()
        rec = self.CT_obj.FDK_op(h)
        t_rec = time.time() - t
        if compute_results:
            self.comp_results(rec, measures, h, filt_name, t_rec)
            rec = None
            gc.collect()
        else:
            return rec

    def filt_LP(self, base_filt, LP_filt, compute_results=True,
                    measures=['MSE', 'MAE', 'SSIM']):
        t = time.time()
        hf = self.FDK_filt(base_filt)
        h = self.CT_obj.pd_IFFT(hf)
        h = sup.low_pass_filter(h, LP_filt)
        rec = self.CT_obj.FDK_op(h)
        t_rec = time.time() - t
        
        if compute_results:
            self.comp_results(rec, measures, h, base_filt + ' + ' + LP_filt[0]
                                + str(LP_filt[1]), t_rec)
        else:
            return rec


    def show_filt(self, rec_it=-1):
        h = np.asarray(self.CT_obj.pd_IFFT(self.results.var[rec_it]))
        self.plot_filt(h)

# %%
class SIRT_class(algorithm_class):
    def __init__(self, CT_obj, non_neg=False):
        self.CT_obj = CT_obj
        if non_neg:    
            self.method = 'SIRT+'
        else:
            self.method = 'SIRT'
        self.non_neg = non_neg

    def do(self, niter, compute_results=True, measures=['MSE', 'MAE', 'SSIM']):
        rec, t_rec = SIRT_meth.SIRT_astra(self.CT_obj.g, niter,
                                          self.CT_obj.geometry,
                                          self.CT_obj.reco_space,
                                          self.CT_obj.WV_path,
                                          self.non_neg)
        
        if type(niter) == list:
            if not compute_results:
                rec_list = []
            for i in range(np.size(niter)):
                rec_arr = self.CT_obj.reco_space.element(np.load(
                                self.CT_obj.WV_path + rec[i]))
                
                if compute_results:
                    self.comp_results(rec_arr, measures, niter[i],
                                      'i=' + str(niter[i]), t_rec[i])
                else:
                    rec_list += [rec_arr]
            if not compute_results:        
                return rec_list
        else:
            rec_arr = self.CT_obj.reco_space.element(np.load(
                            self.CT_obj.WV_path + rec))
            if compute_results:
                self.comp_results(rec_arr, measures, niter, 'i=' + str(niter),
                              t_rec)
            else:
                return rec_arr


# %%
class AFFDK_class(algorithm_class):
    def __init__(self, CT_obj):
        self.CT_obj = CT_obj
        self.method = 'AF-FDK'

    def h(self, rec_it=-1):
        return self.CT_obj.Exp_op(self.results.var[rec_it])

    def show_filt(self, rec_it=-1):
        h = np.asarray(self.CT_obj.Exp_op(self.results.var[rec_it]))
        self.plot_filt(h)
            
    
# %%
class LSFDK_class(AFFDK_class):
    def __init__(self, CT_obj):
        self.CT_obj = CT_obj
        self.method = 'LS-FDK'

    def comp_results(self, rec, measures, var, param, t_rec):
        self.results = results(self, rec, measures, var, param, t_rec)

    def do(self, measures=['MSE', 'MAE', 'SSIM'], compute_results=True):
        t = time.time()
        self.CT_obj.Matrix_Comp_A()
        x = self.CT_obj.Exp_op.domain.element(
                np.linalg.solve(self.CT_obj.AtA, self.CT_obj.Atg))
        rec = self.CT_obj.FDK_bin(x)
        t_rec = time.time() - t
        if compute_results:
            self.comp_results(rec, measures, x, ' ', t_rec)
        else:
            return rec


class TFDK_class(AFFDK_class):
    def __init__(self, CT_obj):
        self.CT_obj = CT_obj
        self.method = 'T-FDK'

    def do(self, lam, measures=['MSE', 'MAE', 'SSIM'], compute_results=True):
        t = time.time()
        self.CT_obj.Matrix_Comp_A()
        if lam == 'optim':
            if hasattr(self, 'optim_lam'):
                pass
            else:
                self.optim_param()
            lam = self.optim_lam
        lamI = lam * self.CT_obj.DDC_norm * np.identity(
                np.shape(self.CT_obj.AtA)[0])
        x = self.CT_obj.Exp_op.domain.element(
                np.linalg.solve(self.CT_obj.AtA + lamI, self.CT_obj.Atg))
        rec = self.CT_obj.FDK_bin(x)
        t_rec = time.time() - t
        if compute_results:
            self.comp_results(rec, measures, x, 'lam=' + str(lam), t_rec)
        else:
            return rec
        

    def optim_param(self, factor=4, it=200):
        # Create low resolution problem
        self.CT_obj.compute_GS(factor, it)
        voxels_LR = np.asarray(self.CT_obj.phantom.voxels) // factor
        g_LR = sup.subsamp_data(self.CT_obj.g, factor)
        if self.CT_obj.phantom.data_type == 'simulated':
            DO = phantom(voxels_LR, self.CT_obj.PH, self.CT_obj.angles,
                                  self.CT_obj.noise, self.CT_obj.src_rad,
                                  self.CT_obj.det_rad, load_data_g=g_LR)
        elif self.CT_obj.phantom.data_type == 'real':
            DO = real_data(g_LR, self.CT_obj.pix_size, self.CT_obj.src_rad,
                           self.CT_obj.det_rad, ang_freq=1, zoom=True,
                           offset=self.CT_obj.offset)
        
        expansion_op = 'linear'
        bin_param = 2
        # Initialize operators and algorithms
        CTo = CT.CCB_CT(DO, data_struct=False)
        # Initialize the algorithms (FDK, SIRT, AFFDK)
        CTo.init_algo()
        # Initialize DDF-FDK algorithm    
        CTo.init_DDF_FDK(bin_param, expansion_op)

        # Do a rough search for the optim param:
        lamTr = np.array([10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])    

        L1D = np.zeros((19))
        tel = 0
        for i in range(len(lamTr)):
            rec = CTo.TFDK.do(lamTr[i], compute_results=False)
            L1D[tel] = sup.L1_distance(rec, self.CT_obj.GS)
            tel += 1
        lmin_T = lamTr[np.argmin(L1D[:len(lamTr)])]
        # Do a fine search for the optim param:
        lamTf = np.array([10, 8, 6, 4, 2, 1, .8, .6, .4, .2, .1]) * lmin_T
        for i in range(len(lamTf)):
            rec = CTo.TFDK.do(lamTf[i], compute_results=False)
            L1D[tel] = sup.L1_distance(rec, self.CT_obj.GS)
            tel += 1
        
        if np.argmin(L1D) <= 7:
            self.optim_lam = lamTr[np.argmin(L1D)]
        else:
            self.optim_lam = lamTf[np.argmin(L1D) - 8]
        DO, CTo, g_LR = None, None, None
        gc.collect()
        return self.optim_lam


class SFDK_class(AFFDK_class):
    def __init__(self, CT_obj):
        self.CT_obj = CT_obj
        self.method = 'S-FDK'

    def do(self, lam, measures=['MSE', 'MAE', 'SSIM'], compute_results=True):
        t = time.time()
        self.CT_obj.Matrix_Comp_A()
        self.CT_obj.Matrix_Comp_C()
        if lam == 'optim':
            if hasattr(self, 'optim_lam'):
                pass
            else:
                self.optim_param()
            lam = self.optim_lam
        lamC = lam * self.CT_obj.DDC_norm / self.CT_obj.gradFDK_norm * \
                    self.CT_obj.CtC
        x = self.CT_obj.Exp_op.domain.element(
                np.linalg.solve(self.CT_obj.AtA + lamC, self.CT_obj.Atg))
        rec = self.CT_obj.FDK_bin(x)
        t_rec = time.time() - t 
        if compute_results:
            self.comp_results(rec, measures, x, 'lam=' + str(lam), t_rec)
        else:
            return rec

    def optim_param(self, factor=4):
        # Create low resolution problem
        self.CT_obj.compute_GS()
        voxels_LR = np.asarray(self.CT_obj.phantom.voxels) // factor
        g_LR = sup.subsamp_data(self.CT_obj.g)
        DO = phantom(voxels_LR, self.CT_obj.PH, self.CT_obj.angles,
                              self.CT_obj.noise, self.CT_obj.src_rad,
                              self.CT_obj.det_rad, load_data_g=g_LR)
        expansion_op = 'linear'
        bin_param = 2
        # Initialize operators and algorithms
        CTo = CT.CCB_CT(DO, data_struct=False)
        # Initialize the algorithms (FDK, SIRT, AFFDK)
        CTo.init_algo()
        # Initialize DDF-FDK algorithm    
        CTo.init_DDF_FDK(bin_param, expansion_op)

        # Do a rough search for the optim param:
        lamTr = np.array([10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])    

        L1D = np.zeros((19))
        tel = 0
        for i in range(len(lamTr)):
            rec = CTo.SFDK.do(lamTr[i], compute_results=False)
            L1D[tel] = sup.L1_distance(rec, self.CT_obj.GS)
            tel += 1
        lmin_T = lamTr[np.argmin(L1D[:len(lamTr)])]
        # Do a fine search for the optim param:
        lamTf = np.array([10, 8, 6, 4, 2, 1, .8, .6, .4, .2, .1]) * lmin_T
        for i in range(len(lamTf)):
            rec = CTo.SFDK.do(lamTf[i], compute_results=False)
            L1D[tel] = sup.L1_distance(rec, self.CT_obj.GS)
            tel += 1
        
        if np.argmin(L1D) <= 7:
            self.optim_lam = lamTr[np.argmin(L1D)]
        else:
            self.optim_lam = lamTf[np.argmin(L1D) - 7]
        DO, CTo, g_LR = None, None, None
        gc.collect()
        return self.optim_lam
    
# %%
class PIFDK_class(AFFDK_class):
    def __init__(self, CT_obj):
        self.CT_obj = CT_obj
        self.method = 'PI-FDK'

    def do(self, lam, measures=['MSE', 'MAE', 'SSIM'], compute_results=True):
        t = time.time()
        self.CT_obj.Matrix_Comp_B()
        if lam == 'optim':
            if hasattr(self, 'optim_lam'):
                pass
            else:
                self.optim_param()
            lam = self.optim_lam
        lamI = lam * self.CT_obj.DC_norm * np.identity(
                np.shape(self.CT_obj.BtB)[0])
        x = self.CT_obj.Exp_op.domain.element(
                np.linalg.solve(self.CT_obj.BtB + lamI, self.CT_obj.Btf))
        rec = self.CT_obj.FDK_bin(x)
        t_rec = time.time() - t
        if compute_results:
            self.comp_results(rec, measures, x, 'lam=' + str(lam), t_rec)
        else:
            return rec

    def optim_param(self, factor=4):
        # Create low resolution problem
#        self.CT_obj.compute_GS()
        voxels_LR = np.asarray(self.CT_obj.phantom.voxels) // factor
        g_LR = sup.subsamp_data(self.CT_obj.g)
        DO = phantom(voxels_LR, self.CT_obj.PH, self.CT_obj.angles,
                              self.CT_obj.noise, self.CT_obj.src_rad,
                              self.CT_obj.det_rad, load_data_g=g_LR)
        expansion_op = 'linear'
        bin_param = 2
        # Initialize operators and algorithms
        CTo = CT.CCB_CT(DO)
        # Initialize the algorithms (FDK, SIRT, AFFDK)
        CTo.init_algo()
        # Initialize DDF-FDK algorithm    
        CTo.init_DDF_FDK(bin_param, expansion_op)

        # Do a rough search for the optim param:
        lamTr = np.array([10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])    

        L1D = np.zeros((19))
        tel = 0
        for i in range(len(lamTr)):
            rec = CTo.PIFDK.do(lamTr[i], compute_results=False)
            L1D[tel] = sup.L1_distance(DO.mask(rec), DO.mask(DO.f))
            tel += 1
        lmin_T = lamTr[np.argmin(L1D[:len(lamTr)])]
        # Do a fine search for the optim param:
        lamTf = np.array([10, 8, 6, 4, 2, 1, .8, .6, .4, .2, .1]) * lmin_T
        for i in range(len(lamTf)):
            rec = CTo.PIFDK.do(lamTf[i], compute_results=False)
            L1D[tel] = sup.L1_distance(DO.mask(rec), DO.mask(DO.f))
            tel += 1
        
        if np.argmin(L1D) <= 7:
            self.optim_lam = lamTr[np.argmin(L1D)]
        else:
            self.optim_lam = lamTf[np.argmin(L1D) - 7]
        DO, CTo, g_LR = None, None, None
        gc.collect()
        return self.optim_lam
# %%
class LRF_FDK_class(algorithm_class):
    def __init__(self, CT_obj):
        self.CT_obj = CT_obj
        self.CT_LR = CT_obj.CT_LR
        self.method = 'AF-FDK'

    def filt_interp(self, h_small):
        size = np.size(h_small)
        xp = np.linspace(0, np.pi, size)
        fp = (np.asarray(h_small))
        size2 = np.size(self.CT_obj.filter_space)
        x = np.linspace(0, np.pi, size2)
        const = self.CT_obj.filter_space.weighting.const / \
                self.CT_LR.filter_space.weighting.const
        return self.CT_obj.filter_space.element(np.interp(x, xp, fp)) * const

    def h(self, rec_it=-1):
        return self.results.var[rec_it]

    def show_filt(self, rec_it=-1):
        self.plot_filt(self.results.var[rec_it])

# %%
class LR_LSFDK_class(LRF_FDK_class):
    def __init__(self, CT_obj):
        super().__init__(CT_obj)
        self.method = 'LR-LS-FDK'

    def comp_results(self, rec, measures, var, param):
        self.results = results(self, rec, measures, var, param)

    def do(self, measures=['MSE', 'MAE', 'SSIM']):
        self.CT_LR.Matrix_Comp_A()
        x = self.CT_LR.Exp_op.domain.element(
                np.linalg.solve(self.CT_LR.AtA, self.CT_LR.Atg))
        h_big = self.filt_interp(self.CT_LR.Exp_op(x))
        rec = self.CT_obj.FDK_op(h_big)
        self.comp_results(rec, measures, h_big, param=' ')


class LR_TFDK_class(LRF_FDK_class):
    def __init__(self, CT_obj):
        super().__init__(CT_obj)
        self.method = 'LR-T-FDK'

    def do(self, lam, measures=['MSE', 'MAE', 'SSIM']):
        self.CT_LR.Matrix_Comp_A()
        lamI = lam * self.CT_LR.DDC_norm * np.identity(
                np.shape(self.CT_LR.AtA)[0])
        x = self.CT_LR.Exp_op.domain.element(
                np.linalg.solve(self.CT_LR.AtA + lamI, self.CT_LR.Atg))
        h_big = self.filt_interp(self.CT_LR.Exp_op(x))
        rec = self.CT_obj.FDK_op(h_big)
        self.comp_results(rec, measures, h_big, param='lam=' + str(lam))

class LR_SFDK_class(LRF_FDK_class):
    def __init__(self, CT_obj):
        super().__init__(CT_obj)
        self.method = 'LR-S-FDK'

    def do(self, lam, measures=['MSE', 'MAE', 'SSIM']):
        self.CT_LR.Matrix_Comp_A()
        self.CT_LR.Matrix_Comp_C()
        lamC = lam * self.CT_LR.DDC_norm / self.CT_LR.gradFDK_norm * \
                self.CT_LR.CtC
        x = self.CT_LR.Exp_op.domain.element(
                np.linalg.solve(self.CT_LR.AtA + lamC, self.CT_LR.Atg))
        h_big = self.filt_interp(self.CT_LR.Exp_op(x))
        rec = self.CT_obj.FDK_op(h_big)
        self.comp_results(rec, measures, h_big, param='lam=' + str(lam))
