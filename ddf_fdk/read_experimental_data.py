#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:56:25 2019

@author: lagerwer
"""

import numpy as np
import os
import pylab
import re
from tqdm import tqdm


# %%
def read_tiff(file, sample = [1, 1]):
    """
    Read a single image.
    """
    im = pylab.imread(file)

    if sample != 1:
        im = im[::sample[0], ::sample[1]]
    
    return im
def get_files_sorted(path, name):
    """
    Sort file entries using the natural (human) sorting
    """
    # Get the files
    files = os.listdir(path)
    
    # Get the files that are alike and sort:
    files = [os.path.join(path, x) for x in files if (name in x)]

    # Keys
    keys = [int(re.findall('\d+', f)[-1]) for f in files]

    # Sort files using keys:
    files = [f for (k, f) in sorted(zip(keys, files))]

    return files 

def read_raw(path, name, sample = [1, 1], dtype = 'float32'):
    """
    Read tiff files stack and return numpy array.
    
    Args:
        path (str): path to the files location
        name (str): common part of the files name
        skip (int): read every so many files
        sample (int): sampling factor in x/y direction
        x_roi ([x0, x1]): horizontal range
        y_roi ([y0, y1]): vertical range
        dtype (str or numpy.dtype): data type to return
        memmap (str): if provided, return a disk mapped array to save RAM
        
    Returns:
        numpy.array : 3D array with the first dimension representing the
        image index
        
    """  
        
    # Retrieve files, sorted by name
    files = get_files_sorted(path, name)
    
    # Read the first file:
    image = read_tiff(files[0], sample)
    sz = np.shape(image)
    
    file_n = len(files)
    
    data = np.zeros((file_n, sz[0], sz[1]), dtype = np.float32)
    
    # Read all files:
    for ii in tqdm(range(file_n)):
        
        filename = files[ii]
        try:
            a = read_tiff(filename, sample)
        except:
            print('WARNING! FILE IS CORRUPTED. CREATING A BLANK IMAGE: ',
                  filename)
            a = np.zeros(data.shape[1:], dtype = np.float32)
         
        #flexUtil.display_slice(a)    
        data[ii, :, :] = a


    return data    