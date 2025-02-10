# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:38:15 2024

@author: gw3013
"""

import numpy as np
import pandas as pd
import os

def create_folder(folder_path: str):
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


# pth_basinlist = "D:/07b_GRL_spinoff/data/CAMELS_US/basins_camels_us_531.txt"
# n_folds=5
# pth_out = "D:/07b_GRL_spinoff/data/folds/"


def create_folds(pth_basinlist, n_folds, pth_out):
    
    seed=478
    
    # set seed
    np.random.seed(seed)
    
    # load basinlist (simple txt file with one row)
    data = pd.read_table(pth_basinlist, header=None).squeeze().tolist()
    data = [str(item).zfill(8) for item in data]
    
    # create folds
    folds = np.random.choice(data, size=(int(len(data)/n_folds), n_folds), 
                             replace=False)
    
    # create fold names
    foldnames = [a + b for a, b in zip(np.repeat("fold",n_folds), 
                                       np.arange(n_folds).astype(str))]
    
    # create output folder (in data directory)
    pth_outfolder = pth_out
    create_folder(folder_path=pth_outfolder)
    
    # output folds in separate files
    for i in range(len(foldnames)):
        pd.DataFrame(np.delete(folds, i, axis=1).flatten()).to_csv(pth_outfolder+foldnames[i]+".txt", 
                                        header=False, index=False)
        pd.DataFrame(folds[:,i]).to_csv(pth_outfolder+foldnames[i]+"_test.txt", 
                                        header=False, index=False)
        
        
#%% execute


create_folds(pth_basinlist = "D:/07b_GRL_spinoff/data/CAMELS_US/basins_camels_us_531.txt",
             n_folds=5, pth_out = "D:/07b_GRL_spinoff/data/folds/")
        
        