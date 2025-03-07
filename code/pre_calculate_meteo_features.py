# -*- coding: utf-8 -*-
"""

@author: Benedikt Heudorfer, March 2025

This code calculates the meteorological static features, i.e. the mean and standard
deviation from the 15 meteorological dynamic features of the paper "Are Deep 
Learning Models in Hydrology Entity Aware?" by Heudorfer et al. (2025).. 

"""

# packages
import os
import pandas as pd
from scipy.stats import skew, kurtosis

# path to camels streamflow time series
pth_camels = "D:/07b_GRL_spinoff/data/CAMELS_US/basin_mean_forcing/all/"

# path to basin selection
pth_basins = "D:/07b_GRL_spinoff/data/CAMELS_US/basins_camels_us_531.txt"

# path to save output results
pth_results = "D:/07b_GRL_spinoff/data/CAMELS_US/basin_mean_forcing/"

# load basin list
basins = pd.read_csv(pth_basins, header=None).squeeze().tolist()
basins = [str(item).zfill(8) for item in basins]

# open container
pths_metdat = []

# get paths of all meteo input files
pth_files = [pth_camels+item for item in os.listdir(pth_camels)]
for i in range(len(pth_files)):
    tempID = pth_files[i].split("/")[-1].split(".")[0]
    if tempID in basins:
        pths_metdat.append(pth_files[i])
del i, pth_files, tempID

    
#%% functions


# Function to calculate mean and standard deviation for a single file
def process_file(file_path, training_period):
    
    # get gauge ID
    gauge_id = file_path.split('/')[-1].split('.')[0]
    
    # read file
    df = pd.read_csv(file_path, skiprows=1)
    
    # restrict to training period
    df['date'] = pd.to_datetime(df[['Year', 'Mnth', 'Day', 'Hr']].astype(str).agg('-'.join, axis=1))
    df.set_index("date", inplace=True)
    df = df.loc[training_period[0]:training_period[1]]    

    # Columns to process
    columns = ['prcp_daymet', 'srad_daymet', 'tmax_daymet', 'tmin_daymet', 'vp_daymet',
               'prcp_maurer', 'srad_maurer', 'tmax_maurer', 'tmin_maurer', 'vp_maurer',
               'prcp_nldas', 'srad_nldas', 'tmax_nldas', 'tmin_nldas', 'vp_nldas']
    
    # Calculate mean, standard deviation, skewness, and kurtosis
    means = df[columns].mean()
    stds = df[columns].std()
    skews = df[columns].apply(lambda x: skew(x, bias=False))
    kurtoses = df[columns].apply(lambda x: kurtosis(x, bias=False))
    
    return [gauge_id] + means.tolist() + stds.tolist() + skews.tolist() + kurtoses.tolist()


#%% calculate meteo static features (execute above function)


# Process all files and collect results
results = [process_file(file, training_period=["1999-10-01", "2008-09-30"]) for file in pths_metdat]

# Define columns for the output
columns = [
    "gauge_id",
    'prcp_daymet_mean', 'srad_daymet_mean', 'tmax_daymet_mean', 'tmin_daymet_mean', 'vp_daymet_mean',
    'prcp_maurer_mean', 'srad_maurer_mean', 'tmax_maurer_mean', 'tmin_maurer_mean', 'vp_maurer_mean',
    'prcp_nldas_mean', 'srad_nldas_mean', 'tmax_nldas_mean', 'tmin_nldas_mean', 'vp_nldas_mean',
    'prcp_daymet_std', 'srad_daymet_std', 'tmax_daymet_std', 'tmin_daymet_std', 'vp_daymet_std',
    'prcp_maurer_std', 'srad_maurer_std', 'tmax_maurer_std', 'tmin_maurer_std', 'vp_maurer_std',
    'prcp_nldas_std', 'srad_nldas_std', 'tmax_nldas_std', 'tmin_nldas_std', 'vp_nldas_std',
    'prcp_daymet_skew', 'srad_daymet_skew', 'tmax_daymet_skew', 'tmin_daymet_skew', 'vp_daymet_skew',
    'prcp_maurer_skew', 'srad_maurer_skew', 'tmax_maurer_skew', 'tmin_maurer_skew', 'vp_maurer_skew',
    'prcp_nldas_skew', 'srad_nldas_skew', 'tmax_nldas_skew', 'tmin_nldas_skew', 'vp_nldas_skew',
    'prcp_daymet_kurt', 'srad_daymet_kurt', 'tmax_daymet_kurt', 'tmin_daymet_kurt', 'vp_daymet_kurt',
    'prcp_maurer_kurt', 'srad_maurer_kurt', 'tmax_maurer_kurt', 'tmin_maurer_kurt', 'vp_maurer_kurt',
    'prcp_nldas_kurt', 'srad_nldas_kurt', 'tmax_nldas_kurt', 'tmin_nldas_kurt', 'vp_nldas_kurt']

# Create a DataFrame from results
results_df = pd.DataFrame(results, columns=columns)

# Save results to csv
results_df.to_csv(pth_results+'/all_sumstat_4.csv', sep=';', index=False)

