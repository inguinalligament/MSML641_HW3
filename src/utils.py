################################################################################################
####        TITLE: MSML HW3                                                                 ####
####        DESCRIPTION: SENTIMENT ANALYSIS - UTILS.PY                                      ####
####        AUTHOR: BRADLEY SCOTT                                                           ####
####        UMD ID: 119 775 028                                                             ####
####        DATE: 26OCT2025                                                                 ####
####        REFERENCES USED (see paper for full details):                                   ####
####            ChatGPT 5                                                                   ####
################################################################################################

'''
[BS10262025] ut3_641_000001
[BS10262025] import all necessary modules
'''
import os, random, time, json
import numpy as np
import tensorflow as tf
import pandas as pd

'''
[BS10262025] ut3_641_000005
[BS10262025] set the seed for reproducibility
'''
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

'''
[BS10262025] ut3_641_000010
[BS10262025] set the timer for epoch timing
'''
def timer(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper

'''
[BS10262025] ut3_641_000015
[BS10262025] build a function to save the results to CSV
'''
def save_results_to_csv(results, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)