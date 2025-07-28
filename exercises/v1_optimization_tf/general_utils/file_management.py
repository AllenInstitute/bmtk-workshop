#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:50:38 2021

@author: jgalvan
"""

import os
import pickle
import lzma
import bz2
import h5py
import gzip
import lz4.frame
# import zstd
import numpy as np
from pathlib import Path



"""
In this module we present different options for saving data (many of them include 
a compression algorithm). The chosen one depends on the particular requirements 
regarding memory, saving and loading time. Here we provide a brief description 
of each:
    1) lzma: is both name of the algorithm and Python module. It can produce 
    higher compression ratio than some older methods and is the algorithm behind 
    the xz utility (more specifically LZMA2). Best method if compression ratio
    is a must, but deadly slow speed of compression and decompression.
    2) h5py: The HDF5 format is a compressed format. The size of all data contained 
    within HDF5 is optimized which makes the overall file size smaller. Even when 
    compressed, however, HDF5 files often contain big data and can thus still be 
    quite large. A powerful attribute of HDF5 is data slicing, by which a particular 
    subsets of a dataset can be extracted for processing. This means that the entire 
    dataset doesn't have to be read into memory (RAM); very helpful in allowing us to 
    more efficiently work with very large (gigabytes or more) datasets!
    3) gzip: is a utility most of us are familiar with. It's also a name of a 
    Python module. This module uses the already mentioned zlib compression 
    algorithm and serves as an interface similar to the gzip and gunzip utilities.
    4) pickle:
    5) bz2: is a module that provides support for bzip2 compression. This 
    algorithm is generally more effective than the deflate method, but might be 
    slower. It also works only on individual files and therefore can't create 
    archives. Used for string data.
    6) lz4: really fast in compression and decompression, however its compression
    ratio is disappointing.
    7) np.savez_compressed: uses zipfile.ZIP_DEFLATED algorithm

We performed a test to illustrate this differences:
data : TYPE, np.ndarray
       SHAPE, (1, 2500, 51978)
       DTYPE, float16

Method                 Memory consumption     Saving time       Loading time
-------------------------------------------------------------------------------
lzma -preset default         67.7 MB            247.9 s             7.5 s
lzma -preset 1              126.5 MB             56.1 s             
lzma -preset 2              115.4 MB             77.8 s             
lzma -preset 3               80.1 MB            130.6 s             
lzma -preset 4               71.2 MB            177.6 s             
h5py                        253.8 MB              4.9 s             0.3 s
gzip                        132.2 MB             21.9 s             2.8 s
pickle                      253.8 MB              3.1 s             0.2 s
bz2                          55.9 MB             27.8 s            11.0 s
np.savez_compressed         132.0 MB             21.0 s             2.2 s
lz4                         203.2 MB              4.7 s             1.5 s
-------------------------------------------------------------------------------

Benchmarks:
https://www.gaia-gis.it/fossil/librasterlite2/wiki?name=benchmarks+(2019+update)
"""

##############################################################################################################################
"""                                                   I. Saving                                                        """
##############################################################################################################################

def save_lzma(data, filename, parent_dir):
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    fn = os.path.join(parent_dir, filename)
    with lzma.LZMAFile(fn+'.lzma', 'wb') as f:
        pickle.dump(data, f)

def save_h5py(data, filename, parent_dir):
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    fn = os.path.join(parent_dir, filename)
    with h5py.File(fn + '.hdf5', 'wb') as f:
        dset = f.create_dataset("default", data=data)
        
def save_gzip(data, filename, parent_dir):
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    fn = os.path.join(parent_dir, filename)
    with gzip.open(fn + '.gzip', 'wb') as f:
        pickle.dump(data, f)
        
def save_pickle(data, filename, parent_dir):
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    fn = os.path.join(parent_dir, filename)
    pikd = open(fn + '.pkl', 'wb')
    pickle.dump(data, pikd)
    pikd.close()
    
# def save_bz2(data, filename, parent_dir):
#     Path(parent_dir).mkdir(parents=True, exist_ok=True)
#     fn = os.path.join(parent_dir, filename)
#     with bz2.BZ2File(fn + '.bz2', 'wb') as f: 
#         pickle.dump(data, f)
        
def save_npz(data, filename, parent_dir):
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    fn = os.path.join(parent_dir, filename)
    np.savez_compressed(fn+'.npz', data)
    
def save_lz4(data, filename, parent_dir):
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    fn = os.path.join(parent_dir, filename)
    with lz4.frame.open(fn+'.lz4', 'wb') as f:
        pickle.dump(data, f)
        
# def save_zstd(data, filename, parent_dir, level=3):
#     Path(parent_dir).mkdir(parents=True, exist_ok=True)
#     compressor = zstd.ZstdCompressor(level=level)
#     with open(filename+'.zstd', 'wb') as f:
#         pickle.dump(compressor.compress(np.ravel(data)), f)
        
        
##############################################################################################################################
"""                                                   II. Loading                                                        """
##############################################################################################################################

def load_h5py(path):
    with h5py.File(path, 'r') as f:
        data = np.array(f['default'])
        return data

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data

def load_gzip(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
        return data

def load_lzma(path):
    with lzma.LZMAFile(path, 'rb') as f:
        data = pickle.load(f)
        return data

# def load_bz2(path):
#     with bz2.open(path, 'rb') as f:
#         data = f.read()
#         return data
    
def load_npz(path):
    data = np.load(path)['arr_0']
    return data
    
def load_lz4(path):
    with lz4.frame.open(path, 'rb') as f:
        data = pickle.load(f)
        return data
    
# def load_zstd(path):
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
#         data = zstd.decompress(data)
#         return data