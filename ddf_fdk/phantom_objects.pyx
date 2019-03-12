#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, nonecheck=False
"""
Created on Fri Jul 13 09:22:03 2018

@author: lagerwer
"""
import numpy as np
from libc.math cimport exp, atan, M_PI, pow, sin, cos
cimport cython
from cython.parallel import prange

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def addGaussianBlob(int size):
    cdef int i, j, k
    cdef double x_c, y_c, z_c, a, theta, phi, x_s, y_s, z_s, st, ct, sf, cf
    cdef double x, y, z

    x_c = (.25 + .5 * np.random.rand()) * size
    y_c = (.25 + .5 * np.random.rand()) * size
    z_c = (.25 + .5 * np.random.rand()) * size
    a = -.25 + 1.25 * np.random.rand()
    theta = 2 * np.pi * np.random.rand()
    phi = np.pi * np.random.rand()
    x_s = (.05 + .1 * np.random.rand()) * size
    y_s = (.05 + .1 * np.random.rand()) * size
    z_s = (.05 + .1 * np.random.rand()) * size
    st = np.sin(theta)
    ct = np.cos(theta)
    sf = np.sin(phi)
    cf = np.cos(phi)
    img = np.zeros((size, size, size))
    cdef double [:, :, :] c_img = img
    for i in prange(size, nogil=True):
        for j in prange(size):
            for k in prange(size):
                x = x_c + (i - x_c) * ct - (j - y_c) * st
                y = y_c + (i - x_c) * cf * st + (j - y_c) * cf * ct - \
                            (k - z_c) * sf
                z = z_c + (i - x_c) * sf * st + (j - y_c) * sf * ct - \
                            (k - z_c) * cf
                c_img[i, j, k] += a * exp(-1/2*(pow(x - x_c, 2) / pow(x_s, 2) \
                                         + pow(y - y_c, 2) / pow(y_s, 2) + \
                                         pow(z - z_c, 2) / pow(z_s, 2)))

    return img


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def addRectangle(int size):
    cdef int i, j, k
    cdef double x_c, y_c, z_c, a, w2, h2, d2, theta, phi, ct, st, cf, sf
    cdef double x, y, z
    x_c = (.25 + .5 * np.random.rand()) * size
    y_c = (.25 + .5 * np.random.rand()) * size
    z_c = (.25 + .5 * np.random.rand()) * size
    a = np.random.rand()
    w2 = (.25 + .5 * np.random.rand()) * size / 2
    h2 = (.0625 + .125 * np.random.rand()) * size / 2
    d2 = (.0625 + .125 * np.random.rand()) * size / 2
    theta = 2 * np.pi * np.random.rand()
    phi = np.pi * np.random.rand()
    ct = np.cos(theta)
    st = np.sin(theta)
    cf = np.cos(phi)
    sf = np.sin(phi)
    img = np.zeros((size, size, size))
    cdef double [:, :, :] c_img = img

    for i in prange(size, nogil=True):
        for j in prange(size):
            for k in prange(size):
                x = (i - x_c) * ct - (j - y_c) * st
                y = (i - x_c) * cf * st + (j - y_c) * cf * ct - \
                            (k - z_c) * sf
                z = (i - x_c) * sf * st + (j - y_c) * sf * ct + \
                            (k - z_c) * cf
                if (x >= -w2) and (x <= w2) and \
                    (y >= -h2) and (y <= h2) and \
                    (z >= -d2) and (z <= d2):
                    c_img[i, j, k] += a
    return img

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def addSiemensStar(int size):
    cdef int i, j, k
    cdef double x_c, y_c, z_c, a, r2, h2, theta, phi, SpokeRange, cf, sf, dx,
    cdef double z, dy, r_pos, t_pos, r_rel
    x_c = (.25 + .5 * np.random.rand()) * size
    y_c = (.25 + .5 * np.random.rand()) * size
    z_c = (.25 + .5 * np.random.rand()) * size
    a = np.random.rand()
    r2 =  ((.0625 + .1 * np.random.rand()) * size ) ** 2
    h2 = (.0625 + .0625 * np.random.rand()) * size / 2
    theta = 2 * np.pi * np.random.rand()
    SpokeRange = 2 * np.pi / 16
    phi = 2 * np.pi * np.random.rand()
    cf = np.cos(phi)
    sf = np.sin(phi)

    img = np.zeros((size, size, size))
    cdef double [:, :, :] c_img = img
    for i in prange(size, nogil=True):
        for j in prange(size):
            for k in prange(size):
                dx = i - x_c
                dy = (j - y_c) * cf - (k - z_c) * sf
                z = (j - y_c) * sf + (k - z_c) * cf
                r_pos = dx ** 2 + dy ** 2
                t_pos = atan(dy / dx) - theta
                r_rel = r_pos / r2 + 0.2 * a
                while t_pos < 0:
                    t_pos = t_pos + M_PI
                if ((t_pos // SpokeRange % 2) == 0) and r_pos <= r2 and \
                        (z >= -h2 * r_rel) and (z <= h2 * r_rel):
                    c_img[i, j, k] += a
    return img



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def addGaussianBlob2D(int size):
    cdef int i, j
    cdef double x_c, y_c, a, theta, x_s, y_s, x_s2, y_s2, c_s2

    x_c = (.25 + .5 * np.random.rand()) * size
    y_c = (.25 + .5 * np.random.rand()) * size
    a =  -.25 + 1.25 * np.random.rand()
    theta = 2 * np.pi * np.random.rand()

    x_s = (.05 + .1 * np.random.rand()) * size
    y_s = (.05 + .1 * np.random.rand()) * size
    x_s2 = np.cos(theta) ** 2 / (2 * x_s ** 2) + \
            np.sin(theta) ** 2 / (2 * y_s ** 2)
    y_s2 = np.sin(theta) ** 2 / (2 * x_s ** 2) + \
            np.cos(theta) ** 2 / (2 * y_s ** 2)
    c_s2 = np.sin(2 * theta) ** 2 / (4 * x_s ** 2) + \
            np.cos(2 * theta) ** 2 / (4 * y_s ** 2)
    img = np.zeros((size, size))
    cdef double [:,:] c_img = img
    for i in range(size):
        for j in range(size):
            c_img[i, j] += a * exp(-((i - x_c) * (i - x_c) * x_s2 +
                                   (j - y_c) * (j - y_c) * y_s2 +
                                   2 * (i - x_c) * (j - y_c) * c_s2))

    return img

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def addRectangle2D(int size):
    cdef int i, j
    cdef double x_c, y_c, a, w2, h2, theta, ct, st, x_rot, y_rot
    x_c = (.25 + .5 * np.random.rand()) * size
    y_c = (.25 + .5 * np.random.rand()) * size
    a = np.random.rand()
    w2 = (.25 + .5 * np.random.rand()) * size / 2
    h2 = (.0125 + .0375 * np.random.rand()) * size / 2
    theta = 2 * np.pi * np.random.rand()
    ct = np.cos(theta)
    st = np.sin(theta)
    img = np.zeros((size, size))
    cdef double [:,:] c_img = img

    for i in range(size):
        for j in range(size):
            x_rot = ct * (i - x_c) - st * (j - y_c)
            y_rot = st * (i - x_c) - ct * (j - y_c)
            if (x_rot >= -w2) and (x_rot <= w2) and \
                (y_rot >= -h2) and (y_rot <= h2):
                    c_img[i, j] += a
    return img

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def addSiemensStar2D(int size):
    cdef int i, j
    cdef double x_c, y_c, a, r2, theta, SpokeRange, dx, dy, r_pos, t_pos
    x_c = (.25 + .5 * np.random.rand()) * size
    y_c = (.25 + .5 * np.random.rand()) * size
    a = np.random.rand()
    r2 = ((.025 + .1 * np.random.rand()) * size ) ** 2
    theta = 2 * np.pi * np.random.rand()
    SpokeRange = 2 * np.pi / 16

    img = np.zeros((size, size))
    cdef double [:,:] c_img = img
    for i in range(size):
        for j in range(size):
            dx = i - x_c
            dy = j - y_c
            r_pos = dx ** 2 + dy ** 2
            t_pos = atan(dy / dx) - theta
            while t_pos < 0:
                t_pos += M_PI
            if ((t_pos // SpokeRange % 2) == 0) and r_pos <= r2:
                c_img[i, j] += a
    return img

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def addBonesFORBILD(int v, double size):
    cdef int i, j, k
    cdef double x_c, x_c1, x_c2, y_c, y_c1, y_c2, y_c3, z_c, z_c1, z_c2, z_c3 
    cdef double z_lim, z_lim1, z_lim2, y_lim, a, a1, a2, b, b1, b2, r1, r2
    cdef double slope, bias
    cdef double x, y, z
    # E14
    x_c = .5 * v
    y_c = (.5 + 3.6 / (2 * size)) * v
    z_c = .5 * v
    z_lim = 0.25 * cos(M_PI / 12) / size / 2 * v
    a = 1.2 / size / 2 * v
    b = 4 / size / 2 * v
    
    # E15
    y_c1 = (.5 + 9.4 / (2 * size)) * v
    a1 = 0.5525561 / size / 2 * v
    b1 = 2 / size / 2 * v
    z_lim1 = 0.2 / size / 2 * v
    
    # E9 
    x_c1 = (0.5 - 4.3 / size / 2) * v
    y_c2 = (0.5 + 6.8 / size / 2) * v
    z_c1 = (0.5 - 1 / size / 2) * v
    a2 = 1.8 / size / 2 * v
    b2 = 0.24 / size / 2 * v
    z_lim2 = 2 / size / 2 * v
    
    # E10
    x_c2 = (0.5 + 4.3 / size / 2) * v
    
    # C16
    y_c3 = (0.5 -11.15 / size /2 ) * v
    z_c2 = (0.5 - 0.2 / size / 2 ) * v
    r1 = (0.5 / size / 2) * v
    r2 = (0.2 / size / 2) * v
    y_lim = .75 / size / 2 * v
    bias = 0.35 / size / 2 * v
    slope = -0.15 / y_lim / size /2 * v
    
    # C17 
    z_c3 = (0.5 + 0.2 / size / 2 ) * v
    ph = np.zeros((v, v, v))
    cdef double [:, :, :] c_ph = ph
    for i in prange(v, nogil=True):
        for j in prange(v):
            for k in prange(v):
                x = i - x_c
                y = (j - y_c) * sin(M_PI / 6) + (k - z_c) * cos(M_PI / 6)
                z = (k - z_c) * sin(M_PI / 6) + (j - y_c) * -cos(M_PI / 6) 
                if z >= -z_lim and z <= z_lim and \
                        pow(x, 2) / pow(a, 2) + pow(y, 2) / pow(b, 2) <= 1:
                    c_ph[i, j, k] = 1.8
                x = (k - z_c) * sin(M_PI / 6) + \
                            (j - y_c1) * cos(M_PI / 6)
                y = (j - y_c1) * -sin(M_PI / 6) + (k - z_c) * cos(M_PI / 6)
                z = i - x_c
                if z >= -z_lim1 and z <= z_lim1 and \
                        pow(x, 2) / pow(a1, 2) + pow(y, 2) / pow(b1, 2) <= 1:
                    c_ph[i, j, k] = 1.8
                x = (i - x_c1) * cos(-M_PI * 5 / 6) + \
                                (j - y_c2) * -sin(-M_PI * 5 / 6)
                y = (j - y_c2) * cos(-M_PI * 5 / 6) + \
                                (i - x_c1) * sin(-M_PI * 5 / 6)
                z = (k - z_c1)
                if z >= -z_lim2 and z <= z_lim2 and \
                        pow(x, 2) / pow(a2, 2) + pow(y, 2) / pow(b2, 2) <= 1:
                    c_ph[i, j, k] = 1.8
                
                x = (i - x_c2) * cos(-M_PI / 6) + \
                            (j - y_c2) * -sin(-M_PI / 6)
                y = (j - y_c2) * cos(-M_PI / 6) + \
                                (i - x_c2) * sin(-M_PI / 6)
                z = (k - z_c1)
                if z >= -z_lim2 and z <= z_lim2 and \
                        pow(x, 2) / pow(a2, 2) + pow(y, 2) / pow(b2, 2) <= 1:
                    c_ph[i, j, k] = 1.8
                x = i - x_c
                y = j - y_c3
                z = k - z_c2
                if y >= -y_lim and y<= y_lim and \
                    pow(x, 2) + pow(z, 2) <= (slope * y + bias) ** 2:
                    c_ph[i, j, k] = 1.8
                
                x = i - x_c
                y = j - y_c3
                z = k - z_c3
                if y >= -y_lim and y<= y_lim and \
                    pow(x, 2) + pow(z, 2) <= (slope * y + bias) ** 2:
                    c_ph[i, j, k] = 1.8
                
    return ph

def addDelta(int size, double a, double sigma):
    cdef int i, j, k
    cdef double x_c, y_c, z_c, x_s, y_s, z_s
    cdef double x, y, z

    x_c = .5 * size
    y_c = .5 * size
    z_c = .5 * size
    x_s = sigma 
    y_s = sigma 
    z_s = sigma 
    img = np.zeros((size, size, size))
    cdef double [:, :, :] c_img = img
    for i in prange(size, nogil=True):
        for j in prange(size):
            for k in prange(size):
                x = i 
                y = j 
                z = k 
                c_img[i, j, k] += a * exp(-1/2*(pow(x - x_c, 2) / pow(x_s, 2) \
                                         + pow(y - y_c, 2) / pow(y_s, 2) + \
                                         pow(z - z_c, 2) / pow(z_s, 2)))

    return img

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def addGaussianBlob_det(int size, double x_c, double y_c, double z_c, double a,
                    double theta, double phi, double x_s, double y_s,
                    double z_s):
    cdef int i, j, k
    cdef double st, ct, sf, cf
    cdef double x, y, z
    x_c *= size
    y_c *= size
    z_c *= size
    x_s *= size
    y_s *= size
    z_s *= size
    st = np.sin(theta)
    ct = np.cos(theta)
    sf = np.sin(phi)
    cf = np.cos(phi)
    img = np.zeros((size, size, size))
    cdef double [:, :, :] c_img = img
    for i in prange(size, nogil=True):
        for j in prange(size):
            for k in prange(size):
                x = x_c + (i - x_c) * ct - (j - y_c) * st
                y = y_c + (i - x_c) * cf * st + (j - y_c) * cf * ct - \
                            (k - z_c) * sf
                z = z_c + (i - x_c) * sf * st + (j - y_c) * sf * ct - \
                            (k - z_c) * cf
                c_img[i, j, k] += a * exp(-1/2*(pow(x - x_c, 2) / pow(x_s, 2) \
                                         + pow(y - y_c, 2) / pow(y_s, 2) + \
                                         pow(z - z_c, 2) / pow(z_s, 2)))

    return img


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def addRectangle_det(int size, double x_c, double y_c, double z_c, double a,
                 double w2, double h2, double d2, double theta, double phi):
    cdef int i, j, k
    cdef double ct, st, cf, sf
    cdef double x, y, z
    x_c *= size
    y_c *= size
    z_c *= size
    w2 *= size / 2
    h2 *= size / 2
    d2 *= size / 2
    ct = np.cos(theta)
    st = np.sin(theta)
    cf = np.cos(phi)
    sf = np.sin(phi)
    img = np.zeros((size, size, size))
    cdef double [:, :, :] c_img = img

    for i in prange(size, nogil=True):
        for j in prange(size):
            for k in prange(size):
                x = (i - x_c) * ct - (j - y_c) * st
                y = (i - x_c) * cf * st + (j - y_c) * cf * ct - \
                            (k - z_c) * sf
                z = (i - x_c) * sf * st + (j - y_c) * sf * ct + \
                            (k - z_c) * cf
                if (x >= -w2) and (x <= w2) and \
                    (y >= -h2) and (y <= h2) and \
                    (z >= -d2) and (z <= d2):
                    c_img[i, j, k] += a
    return img

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def addSiemensStar_det(int size, double x_c, double y_c, double z_c, double a,
                   double r2, double h2, double theta, double phi):
    cdef int i, j, k
    cdef double SpokeRange, cf, sf, dx
    cdef double z, dy, r_pos, t_pos, r_rel
    x_c *= size
    y_c *= size
    z_c *= size
    r2 = (r2 * size) ** 2
    h2 *= size / 2
    SpokeRange = 2 * np.pi / 16
    cf = np.cos(phi)
    sf = np.sin(phi)

    img = np.zeros((size, size, size))
    cdef double [:, :, :] c_img = img
    for i in prange(size, nogil=True):
        for j in prange(size):
            for k in prange(size):
                dx = i - x_c
                dy = (j - y_c) * cf - (k - z_c) * sf
                z = (j - y_c) * sf + (k - z_c) * cf
                r_pos = dx ** 2 + dy ** 2
                t_pos = atan(dy / dx) - theta
                r_rel = r_pos / r2 + 0.2 * a
                while t_pos < 0:
                    t_pos = t_pos + M_PI
                if ((t_pos // SpokeRange % 2) == 0) and r_pos <= r2 and \
                        (z >= -h2 * r_rel) and (z <= h2 * r_rel):
                    c_img[i, j, k] += a
    return img


