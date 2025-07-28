#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:30:24 2021

@author: jgalvan
"""

# Let's try to code a decorator for timing any function:
from functools import wraps
#import numba as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from pathlib import Path
from collections.abc import Iterable
from functools import reduce
import time
from datetime import datetime
import tracemalloc
import subprocess


##############################################################################################################################
"""                                                   I. Decorators                                                        """
##############################################################################################################################

def timer(orig_func):
    """Wrapper for timing functions"""
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        t1 = time.time()
        result = orig_func(*args,**kwargs)
        dt = time.time() - t1
        print('{} ran in: {:.2f} s = {:.2f} min = {:.2f} h'.format(orig_func.__name__, dt,dt/60,dt/3600))
        return result
    return wrapper

class timer_class(object):
    """Wrapper for timing functions. Same as timer but in a class."""
    def __init__(self, orig_func):
        self.orig_func = orig_func

    def __call__(self, *args, **kwargs):
        t1 = time.time()
        result = self.orig_func(*args, **kwargs)
        dt = time.time() - t1
        print('{} ran in: {} sec'.format(self.orig_func.__name__, dt))
        return result

def memory_tracer(orig_func):
    """Wrapper for memory consumption tracing"""
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        # starting the monitoring of memory consumption
        tracemalloc.start()
        result = orig_func(*args,**kwargs)
        # displaying the memory
        current, peak = tracemalloc.get_traced_memory()
        print(f'Current memory [GB]: {current/1024**3}, Peak memory [GB]: {peak/1024**3}')
        # stopping the library
        tracemalloc.stop()
        # La libreria psutil nos permite ver la informacion de memoria de todo el procesador CPU
        # lo cual incluye cores que no estamos usando.... Hay dos tipos de CPU en nuredduna:
        # AMD Epyc2 7402 (48 cores) e Intel Xeon E5-2630 (8 cores)
        
        # import psutil
        # mem_usage = psutil.virtual_memory()
        # print(f"Free: {mem_usage.percent}%")
        # print(f"Total: {mem_usage.total/(1024**3):.2f}GB")
        # print(f"Used: {mem_usage.used/(1024**3):.2f}GB")
        # per_cpu = psutil.cpu_percent(percpu=True)
        # # For individual core usage with blocking, psutil.cpu_percent(interval=1, percpu=True)
        # for idx, usage in enumerate(per_cpu):
        #     print(f"CORE_{idx+1}: {usage}%")
        return result
    return wrapper

##############################################################################################################################
"""                                                 II. Getopt utils                                                       """
##############################################################################################################################

def encoder(x, ndigits=2, iterables=(list, tuple, np.ndarray)):
    """x -> string version of x"""
    if x is None:
        return "none"
    elif isinstance(x, datetime):
        return x.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(x, str):
        return x.replace("/", "*")
    elif isinstance(x, float):
        if x == int(x):
            return str(int(x))
        else:
            return str(round(x, ndigits=ndigits)).replace('.', '--')
    elif isinstance(x, int):
        return str(x)
    elif isinstance(x, iterables):
        return '-'.join([encoder(sub_x) for sub_x in x])
    else:
        return str(x)
    
def decoder(x, iterables=(list, tuple, np.ndarray)):
    """string version of x -> x"""
    if x.lower() == "none":
        return None
    elif x.lower() == "false":
        return False
    elif x.lower() == "true":
        return True
    elif "*" in x:
        return x.replace("*", "/")
    elif "--" in x:
        return float(x.replace("--", "."))
    elif isinstance(x, iterables):
        return [decoder(sub_x) for sub_x in x]
    else:
        try:
            return int(x)
        except:
            return x

def getopt_printer(opts):
    """Prints getopt input in a readable way."""
    print('\n'.join(f'{opt} => {arg}' for opt, arg in (("Args", "Values"), *opts)))
    
def dict_to_id(d, ndigits=2):
    """Generate ID of the form k1-v1_k2-v2... for k_i, v_i keys and values of the dictionary d."""
    key_formatter = lambda k: k.replace("_", "-")
    return "_".join([f"{key_formatter(k)}-{encoder(d[k], ndigits=ndigits)}" for k in sorted(d.keys())])

def id_to_dict(identifier):
    """Inverse of dict_to_id."""
    s = identifier.split("/")[-1] # retain filename only
    s = os.path.splitext(s)[0] # remove extension
    d = {}
    for split in s.split("_"):
        var_value = split.split("-")
        if len(var_value) > 1:
            if "" in var_value: # value is a float
                var_value_arr = np.array(var_value)
                idx_dot = np.argwhere(var_value_arr == "")[0, 0]
                key_idx = 0 if idx_dot == 2 else slice(0, idx_dot-2)
                d["-".join(var_value[key_idx])] = decoder(f"{var_value_arr[idx_dot-1]}--{var_value_arr[idx_dot+1]}")
            else:
                d["-".join(var_value[:-1])] = decoder(var_value[-1])
    return d

def id_updater(filename, update_dict, mode="add"):
    """
    Modifies filename by updating the underlying dict.
    Attrs:
        - filename:    id to be modified
        - update_dict: dict to use for updating the id. if update_dict={} => filename rearranged according to other_utils.dict_to_id.
        - mode:        - "add":    add update_dict to the id.
                       - "delete": delete update_dict from the id.
    Returns modified filename.
    """
    split_dirs = filename.split("/")
    parentDir = "/".join(split_dirs[:-1])
    file = split_dirs[-1]
    d = id_to_dict(file)
    if mode == "add":
        d.update(update_dict)
    elif mode == "delete":
        d = {k: v for k, v in d.items() if k not in update_dict.keys()}
    var_values = [part.split("-") for part in file.split("_")]
    init = "_".join([part[0] for part in var_values if len(part) == 1])
    ext = os.path.splitext(file)[1] 
    new_filename = os.path.join(parentDir, f"{init}_{dict_to_id(d)}{ext}")

    return new_filename

def id_renamer(update_dict, parentDir, key=None, mode="add"):
    """
    Modifies id of files in parentDir by updating the underlying dict.
    Attrs:
        - update_dict: dict to use for updating the id
        - parentDir: folder where files are located.
        - key: string contained in the file for it to be modified.
        - mode:   - "add": add update_dict to the id.
                  - "delete": delete update_dict from the id.
    Returns #modified files.
    NOTE: If update_dict={} => filenames will be rearranged according to other_utils.dict_to_id.
    """
    r = 0
    for file in os.listdir(parentDir):
        if key is None or key in file:
            old_filename = os.path.join(parentDir, file)
            new_filename = id_updater(old_filename, update_dict, mode=mode)
            os.rename(old_filename, new_filename)
            r += 1
    return r


##############################################################################################################################
"""                                                    III. Other                                                          """
##############################################################################################################################


def latex_table(df, index=False, **kwargs):
    """Pandas DataFrame -> Latex table."""
    col_format = "c" if isinstance(df, pd.core.series.Series) else "c"*len(df.columns)
    if index:
        col_format += "c"
    table_replacements = (("\\toprule", "\\toprule "*2),
                          ("\\bottomrule", "\\bottomrule "*2)
    )
    text_replacements = (("\\textbackslash ", "\\"),
                         ("\{", "{"), 
                         ("\}", "}"),
                         ("\$", "$"),
                         ("\_", "_"),
                         ("\\textasciicircum ", "^")
    )
    table_formatter = lambda x:  reduce(lambda a, kv: a.replace(*kv), table_replacements, x)
    text_formatter = lambda x: reduce(lambda a, kv: a.replace(*kv), text_replacements, x)
    formatter = lambda x: text_formatter(table_formatter(x))
    print(formatter(df.to_latex(index=index, column_format=col_format, **kwargs)))
    return

def fig_saver(filename):
    """Figure saver without overwriting."""
    i = 0
    while os.path.exists('{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig('{}{:d}.png'.format(filename, i))


##############################################################################################################################
"""                                                   I. GPU and CPU memory profilers                                                        """
##############################################################################################################################

import tensorflow as tf

class GPUMemoryTracker:
    def __init__(self):
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE, encoding='utf-8') # MiB
        self.previous_used = float(result.stdout.strip())
    
    def get_gpu_memory(self):
        # Function to get the allocated, free and total memory of a GPU
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE, encoding='utf-8') # MiB
        used, free, total = [float(x) for x in result.stdout.strip().split(',')]
        
        increase = used - self.previous_used
        self.previous_used = used
        
        tf.print("---- GPU Memory ----")
        tf.print(f"  Total: {round(total / 1024, 2)} GiB")
        tf.print(f"  Available: {round(free / 1024, 2)} GiB")
        tf.print(f"  Used: {round(used / 1024, 2)} GiB")
        tf.print(f"  Increase: {round(increase / 1024, 2)} GiB")
        tf.print('')


def print_gpu_memory():
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPU devices found.")
        return

    for i, gpu in enumerate(physical_devices):
        # print(f"GPU {i}: {gpu.name}")
        # Get the GPU name (takes on the form of '/device:GPU:0')
        name = gpu.name
        # Remove the '/device:' prefix and trailing GPU number to get the name
        name = ':'.join(name.split(':')[-2:])

        # Get GPU memory usage
        try:
            memory_details = tf.config.experimental.get_memory_usage(name) # this only gives the memory that Tf is currently using, not all the memory that it has allocated
            # print gpu memory in GB
            print("GPU memory:", memory_details/1024**3, " GB")
        except:
            print("Cannot get memory details for GPU", name)

# def print_system_memory():
#     svmem = psutil.virtual_memory()
#     print(f"System Memory:")
#     print(f"  Total: {svmem.total // (1024 ** 3)} GB")
#     print(f"  Available: {svmem.available // (1024 ** 3)} GB")
#     print(f"  Used: {svmem.used // (1024 ** 3)} GB")
