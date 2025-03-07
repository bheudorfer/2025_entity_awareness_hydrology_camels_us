# -*- coding: utf-8 -*-
"""

@author: Benedikt Heudorfer, March 2025

This code contains the auxiliary mutual information analysis described in section 2.2 of 
the paper "Are Deep Learning Models in Hydrology Entity Aware?" by Heudorfer et al. (2025).

The purpose of this pairwise mutual information analysis is, quote, "to analyse 
the degree of shared information between the static features from the CAMELS dataset 
and the static features derived summary statistics from dynamic features".

"""

# packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

# paths
pth0 = "D:/07b_GRL_spinoff/"
pth_meteo_stat = pth0+"data/CAMELS_US/basin_mean_forcing/"
pth_camels_stat = pth0+"data/CAMELS_US/camels_attributes_v2.0/"
pth_basins = pth0+"data/CAMELS_US/basins_camels_us_531.txt"

# load list of basin ids
basins = pd.read_csv(pth_basins, header=None, dtype=str).squeeze().to_list()


#%% read CAMELS and METEO static attributes


# read METEO
stat_meteo = pd.read_csv(pth_meteo_stat+"all_sumstat_2.csv", sep=";")
stat_meteo.set_index("gauge_id", drop=True, inplace=True)
stat_meteo = stat_meteo.reindex(sorted(stat_meteo.columns), axis=1)

# static attributes that will be used
static_input = ["elev_mean", 
                "slope_mean",
                "area_gages2",
                "frac_forest",
                "lai_max",
                "lai_diff",
                "gvf_max",
                "gvf_diff",
                "soil_depth_pelletier",
                "soil_depth_statsgo",
                "soil_porosity",
                "soil_conductivity",
                "max_water_content",
                "sand_frac",
                "silt_frac",
                "clay_frac",
                "carbonate_rocks_frac",
                "geol_permeability",
                "p_mean",
                "pet_mean",
                "aridity",
                "frac_snow",
                "high_prec_freq",
                "high_prec_dur",
                "low_prec_freq",
                "low_prec_dur"]

# list of file paths to attribute tables
pth_camels_files = [pth_camels_stat+item for item in os.listdir(pth_camels_stat) \
                    if "camels_" in item and ".txt" in item]

# Read one by one the attributes files
stat_camels = []
for file in pth_camels_files:
    temp = pd.read_csv(file, sep=';', header=0, dtype={'gauge_id': str})
    temp = temp.set_index('gauge_id')
    stat_camels.append(temp)
stat_camels = pd.concat(stat_camels, axis=1)
del file, temp

# Filter attributes and basins of interest
stat_camels = stat_camels.loc[basins, static_input]


#%% mutual information


# Function to discretize data using binning
def discretize_data(data, n_bins=10):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretized_data = est.fit_transform(data)
    return discretized_data

# Function to calculate mutual information between two sets of variables
def calculate_mutual_information(set1, set2):
    mi_matrix = np.zeros((set1.shape[1], set2.shape[1]))
    
    for i in range(set1.shape[1]):
        for j in range(set2.shape[1]):
            mi_matrix[i][j] = mutual_info_score(set1[:, i], set2[:, j])
    
    return mi_matrix

# Discretize the data sets
stat_camels_discretized = discretize_data(stat_camels)
stat_meteo_discretized = discretize_data(stat_meteo)

# Calculate mutual information matrix with discretized data
mi_matrix_discretized = calculate_mutual_information(stat_camels_discretized, 
                                                      stat_meteo_discretized)

# Create row and column labels
row_labels = list(stat_camels.columns)
col_labels = list(stat_meteo.columns)

# Plot the heatmap using matplotlib
plt.figure(figsize=(8, 7))
heatmap = plt.imshow(mi_matrix_discretized, aspect='auto', cmap='viridis')
cbar = plt.colorbar(heatmap)
cbar.set_label('Mutual Information Score')  # Add legend title
plt.xticks(ticks=np.arange(len(col_labels)), labels=col_labels, rotation=90)
plt.yticks(ticks=np.arange(len(row_labels)), labels=row_labels)
plt.tight_layout()
plt.savefig(pth0+"plots/mutual_info_heatmap.pdf")
# plt.show()

