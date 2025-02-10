# -*- coding: utf-8 -*-
"""

Created on 19.09.2024
@author: Benedikt Heudorfer


Code to unpack and subsequently summarize the scores of the benchmark model 
from kratzert 2021 into a single csv for easy plotting later on 

The input data o this script is not included in the repository, but can be 
downloaded from https://doi.org/10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1

In there, the results of the kratzert 2021 study are stored in pickle format, which 
are not long-term stable and require these specfic packages to open:
conda create --name pickler python=3.7.1
conda activate pickler
conda install pandas=1.3.5
conda install xarray=0.13.0

"""

import os 
import pickle 
import copy
import pandas as pd 
import numpy as np

pth0 = "D:/07b_GRL_spinoff/data/benchmarks/"

def NSEcalc(observed, simulated):
    nse_score = 1-np.sum((observed-simulated)**2)/np.sum((observed-np.mean(observed))**2)
    return(nse_score)

def KGEcalc(observed, simulated):
    obs_mean = np.mean(observed)
    sim_mean = np.mean(simulated)
    r = np.corrcoef(observed, simulated)[0, 1]  # Pearson correlation coefficient
    beta = sim_mean / obs_mean  # Ratio of means
    gamma = np.std(simulated) / np.std(observed)  # Ratio of standard deviations
    kge_score = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return(kge_score)


#%% unpack original test_period.p


seeds = [item.split("_")[-1].split(".")[0] for item in os.listdir(pth0+"kratzert2021") if "test_results_" in item]
for i in range(len(seeds)):
    seedi = seeds[i]

    with open(pth0+"kratzert2021/test_results_"+seedi+".p", "rb") as file:
        temp = pickle.load(file)

    keyz = temp.keys()
    for key in keyz:
        temp2 = temp[key]['xr']
        temp2 = temp2.to_dataframe()
        temp2.to_csv(pth0+"kratzert2021/all_test_results/"+key+"_"+seedi+".csv")


#%% summarize the scores of kratzert 2021 into a single csv for easy plotting


all_files = os.listdir(pth0+"kratzert2021/all_test_results/")
IDs = np.unique([item.split("_")[0] for item in all_files]).tolist()
seeds = np.unique([item.split("_")[1].split(".")[0] for item in all_files]).tolist()

NSE = pd.DataFrame(columns=["gauge_id","seed","MEDIAN"])
KGE = pd.DataFrame(columns=["gauge_id","seed","MEDIAN"])

for i in range(len(all_files)):

    temp = pd.read_csv(pth0+"kratzert2021/all_test_results/"+all_files[i])
    
    IDi = all_files[i].split("_")[0]
    seedi = all_files[i].split("_")[1].split(".")[0][4:]
    NSEi = NSEcalc(temp["QObs(mm/d)_obs"], temp["QObs(mm/d)_sim"])
    KGEi = KGEcalc(temp["QObs(mm/d)_obs"], temp["QObs(mm/d)_sim"])
    
    NSE_update_df = pd.DataFrame([[IDi,seedi, NSEi]], columns=["gauge_id","seed","MEDIAN"])
    NSE_update_df = pd.DataFrame({"gauge_id":[IDi],"seed":seedi,"MEDIAN":NSEi})
    NSE.loc[len(NSE)] = NSE_update_df.iloc[0,:]
    
    KGE_update_df = pd.DataFrame([[IDi,seedi, KGEi]], columns=["gauge_id","seed","MEDIAN"])
    KGE_update_df = pd.DataFrame({"gauge_id":[IDi],"seed":seedi,"MEDIAN":KGEi})
    KGE.loc[len(KGE)] = KGE_update_df.iloc[0,:]

NSEwide = NSE.pivot(index = "gauge_id", columns = "seed", values = "MEDIAN")
KGEwide = KGE.pivot(index = "gauge_id", columns = "seed", values = "MEDIAN")

NSEwide["MEDIAN"] = NSEwide.median(axis=1)
KGEwide["MEDIAN"] = KGEwide.median(axis=1)

NSEout = NSEwide[["MEDIAN"]]
KGEout = KGEwide[["MEDIAN"]]
NSEout.to_csv(pth0+"NSE_kratzert2021.csv")
KGEout.to_csv(pth0+"KGE_kratzert2021.csv")


#%% check score how kratzert calculated it (mean over 10 hydrographs, created by 
# the seed realizations, instead of median over 10 NSEs calculated from every 
# seed realization hydrograph); exemplary for NSE


all_files = os.listdir(pth0+"kratzert2021/all_test_results/")
IDs = np.unique([item.split("_")[0] for item in all_files]).tolist()
seeds = np.unique([item.split("_")[1].split(".")[0] for item in all_files]).tolist()

NSE = pd.DataFrame(columns=["gauge_id","MEAN"])

for i in range(len(IDs)):
    
    if i%50==0: print(i)

    seed_realizations = [item for item in all_files if IDs[i] in item]
    
    for j in range(len(seed_realizations)):
    
        temp = pd.read_csv(pth0+"kratzert2021/all_test_results/"+seed_realizations[j])
        
        if j==0:
            out = copy.deepcopy(temp)
            out.rename(columns={out.columns[-1]: out.columns[-1]+"_"+str(j)}, inplace=True)
        
        if j!=0:
            out = pd.merge(out,temp)
            out.rename(columns={out.columns[-1]: out.columns[-1]+"_"+str(j)}, inplace=True)
    
    out_NSE = NSEcalc(out["QObs(mm/d)_obs"], out.filter(like='sim').mean(axis=1))
    
    update_df = pd.DataFrame({"gauge_id":[IDs[i]],"MEAN":out_NSE})
    
    NSE.loc[len(NSE)] = update_df.iloc[0,:]

print("NSE median: "+str(NSE.MEAN.median())) # 0.821 like reported in their paper (kratzert 2021)
print("NSE mean: "+str(NSE.MEAN.mean())) # 0.783 like reported in their paper (kratzert 2021)
