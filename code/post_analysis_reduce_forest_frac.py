# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 08:33:34 2025

@author: Benedikt Heudorfer
"""



#%% packages & paths

# packages
import os
import copy
import pickle
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# paths
pth0 = "D:/07b_GRL_spinoff/"
pthres = pth0+"results/"
pth_basinlist = pth0+"data/CAMELS_US/basins_camels_us_63_zhang_forest_reduced.txt"
# pth_out = pth0+"submit_revision/forest_reduction_experiment/"


#%% reduce forest fraction for applicable static features (see supplement text S4)


attr_vege = pd.read_csv(pth0+"data/CAMELS_US/camels_attributes_v2.0/camels_vege.txt", sep=";", dtype={0: str})
basins = pd.read_csv(pth0+"data/CAMELS_US/basins_camels_us_63_zhang_forest_reduced.txt", header=None,dtype=str).squeeze().to_list()

is_zhang_selection = list(np.where(np.isin(attr_vege.gauge_id, basins))[0])

resulting_attr_vege = copy.deepcopy(attr_vege)
resulting_attr_vege.loc[is_zhang_selection,"frac_forest"] = resulting_attr_vege.loc[is_zhang_selection,"frac_forest"]*0.5
resulting_attr_vege.loc[is_zhang_selection,"lai_max"] = resulting_attr_vege.loc[is_zhang_selection,"lai_max"]*0.82
resulting_attr_vege.loc[is_zhang_selection,"gvf_max"] = resulting_attr_vege.loc[is_zhang_selection,"gvf_max"]*0.82

resulting_attr_vege = resulting_attr_vege[["gauge_id","frac_forest","lai_max","gvf_max"]]
resulting_attr_vege = resulting_attr_vege.rename(columns={"frac_forest":"frac_forest_reduced",
                                                          "lai_max":"lai_max_reduced",
                                                          "gvf_max":"gvf_max_reduced"})

resulting_attr_vege.to_csv(pth0+"data/CAMELS_US/camels_attributes_v2.0/camels_vege_forest_reduced.txt", sep=";", index=False)


#%%  get NSEs


# folder list of full model runs
folders_full = [pthres+item for item in os.listdir(pthres) if "full_fold" in item]

# folder list of reduced forest inference runs
folders_forest_reduced = [pthres+item for item in os.listdir(pthres) if "reducedforest" in item]

# list of basins where forest was reduced
basins = pd.read_csv(pth_basinlist, header=None, dtype=str).squeeze().to_list()

# get all NSE of inference runs with reduced forest cover
for i in range(len(folders_forest_reduced)):
    if i==0:
        NSE_reduced = pd.read_csv(folders_forest_reduced[i]+"/NSE.csv")
        continue
    NSE_reduced = pd.concat([NSE_reduced, pd.read_csv(folders_forest_reduced[i]+"/NSE.csv")])

# reformat
NSE_reduced = NSE_reduced.sort_values(by="basin_id").reset_index(drop=True)
NSE_reduced.basin_id = [str(item).zfill(8) for item in NSE_reduced.basin_id]

# find the ones where there is reduced forest
# (only 54 of the original 63 are in the 530 subselection)
keep = [index for index, item in enumerate(NSE_reduced.basin_id.to_list()) if item in basins]
NSE_reduced = NSE_reduced.loc[keep]

#-----------------

# get all NSE of the full model runs with normal forest cover
for i in range(len(folders_full)):
    if i==0:
        NSE = pd.read_csv(folders_full[i]+"/NSE.csv")
        continue
    NSE = pd.concat([NSE, pd.read_csv(folders_full[i]+"/NSE.csv")])

# reformat
NSE = NSE.sort_values(by="basin_id").reset_index(drop=True)
NSE.basin_id = [str(item).zfill(8) for item in NSE.basin_id]

# find the ones where there is reduced forest
# (only 54 of the original 63 are in the 530 subselection)
keep = [index for index, item in enumerate(NSE.basin_id.to_list()) if item in basins]
NSE = NSE.loc[keep]


#%% get sim results


for i in range(len(folders_forest_reduced)):
    if i==0:
        # load the test results
        with open(folders_forest_reduced[i]+"/test_results.pickle", "rb") as input_file:
            temp = pickle.load(input_file)
        # get only the ones where there is reduced forest
        temp = {k: temp[k] for k in basins if k in temp}
        temp = {key + "_"+folders_forest_reduced[i][-3:]: value for key, value in temp.items()}
        # calculate summary statistics
        summary_results_reduced = pd.DataFrame({"basin_id": list(temp.keys()),
                                        "avg":[value.y_sim.median() for key, value in temp.items()
                                                ]})
        continue
    # repreat for other i
    with open(folders_forest_reduced[i]+"/test_results.pickle", "rb") as input_file:
        temp = pickle.load(input_file)
    temp = {k: temp[k] for k in basins if k in temp}
    temp = {key + "_"+folders_forest_reduced[i][-3:]: value for key, value in temp.items()}
    summary_results_reduced = pd.concat([summary_results_reduced,
                                 pd.DataFrame({"basin_id": list(temp.keys()),
                                               "avg":[value.y_sim.median() for key, value in temp.items()]
                                               })])
del temp, input_file

# reformat
summary_results_reduced = summary_results_reduced.sort_values(by="basin_id")
summary_results_reduced.reset_index(drop=True, inplace=True)


#-----------------

for i in range(len(folders_full)):
    if i==0:
        # load the test results
        with open(folders_full[i]+"/test_results.pickle", "rb") as input_file:
            temp = pickle.load(input_file)
        # get only the ones where there is reduced forest
        temp = {k: temp[k] for k in basins if k in temp}
        temp = {key + "_"+folders_forest_reduced[i][-3:]: value for key, value in temp.items()}
        # calculate summary statistics
        summary_results = pd.DataFrame({"basin_id": list(temp.keys()),
                                        "avg":[value.y_sim.median() for key, value in temp.items()
                                                ]})
        continue
    # repreat for other i
    with open(folders_full[i]+"/test_results.pickle", "rb") as input_file:
        temp = pickle.load(input_file)
    temp = {k: temp[k] for k in basins if k in temp}
    temp = {key + "_"+folders_forest_reduced[i][-3:]: value for key, value in temp.items()}
    summary_results = pd.concat([summary_results,
                                 pd.DataFrame({"basin_id": list(temp.keys()),
                                               "avg":[value.y_sim.median() for key, value in temp.items()]
                                               })])
del temp, input_file

# reformat
summary_results = summary_results.sort_values(by="basin_id")
summary_results.reset_index(drop=True, inplace=True)


#%% comparison table


seeds = [227, 310, 325, 514, 550]

final_result = pd.DataFrame({"seed":seeds+["all"],
                             "Q_diff_avg_perc":None,
                             "perc_incr":None,
                             "perc_decr":None,
                             "t_stat":None,
                             "p_val":None})

for i in range(len(seeds)):

    sample = summary_results[summary_results['basin_id'].str.contains(str(seeds[i]))]
    sample_reduced = summary_results_reduced[summary_results_reduced['basin_id'].str.contains(str(seeds[i]))]
    
    t_statistic, p_value = stats.ttest_rel(sample.avg, sample_reduced.avg)
    
    final_result.loc[i,"Q_diff_avg_perc"] = round((sample_reduced.avg.mean()/sample.avg.mean()-1)*100,1)
    final_result.loc[i,"perc_incr"] = round(sum((sample_reduced.avg-sample.avg)>0)/len(sample)*100)
    final_result.loc[i,"perc_decr"] = round(sum((sample_reduced.avg-sample.avg)<0)/len(sample)*100)
    final_result.loc[i,"t_stat"] = round(t_statistic,3)
    final_result.loc[i,"p_val"] = p_value

t_statistic, p_value = stats.ttest_rel(summary_results.avg, summary_results_reduced.avg)

final_result.loc[i+1,"Q_diff_avg_perc"] = round((summary_results_reduced.avg.mean()/summary_results.avg.mean()-1)*100,1)
final_result.loc[i+1,"perc_incr"] = round(sum((summary_results_reduced.avg-summary_results.avg)>0)/len(summary_results)*100)
final_result.loc[i+1,"perc_decr"] = round(sum((summary_results_reduced.avg-summary_results.avg)<0)/len(summary_results)*100)
final_result.loc[i+1,"t_stat"] = round(t_statistic,3)
final_result.loc[i+1,"p_val"] = p_value

final_result.to_excel(pthres+"t_test.xlsx", index=False)

