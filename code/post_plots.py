# -*- coding: utf-8 -*-
"""

@author: Benedikt Heudorfer, March 2025

This code reproduces all figures (except mutual information figure S2) found in the 
paper "Are Deep Learning Models in Hydrology Entity Aware?" by Heudorfer et al. (2025).

"""

# packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geopandas as gpd
from scipy import stats
from matplotlib.colors import ListedColormap
import pickle
import copy

# paths
pth_models = "D:/07b_GRL_spinoff/results/"
pth_basins = "D:/07b_GRL_spinoff/data/CAMELS_US/basins_camels_us_531.txt"
pth_coords = "D:/07b_GRL_spinoff/data/CAMELS_US/camels_attributes_v2.0/camels_topo.txt"
pth_usashape = "D:/07b_GRL_spinoff/data/shapes/cb_2018_us_nation_20m.shp"
pth_benchmarks = "D:/07b_GRL_spinoff/data/benchmarks/"
pth_plot = "D:/07b_GRL_spinoff/plots/"



#%% functions


# Function to calculate NSE, R2, and KGE
def calculate_scores(observed, simulated):
    
    # NSE
    ss_res = np.sum((observed - simulated) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    nse_score = 1 - (ss_res / ss_tot)
    
    # KGE
    obs_mean = np.mean(observed)
    sim_mean = np.mean(simulated)
    r = np.corrcoef(observed, simulated)[0, 1]  # Pearson correlation coefficient
    beta = sim_mean / obs_mean  # Ratio of means
    gamma = np.std(simulated) / np.std(observed)  # Ratio of standard deviations
    kge_score = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

    return nse_score, kge_score, r, beta, gamma

#--------------------------------------------------

# Bootstrapping function
def bootstrap(data, num_iterations=100, sample_fraction=0.8):
    n = int(len(data) * sample_fraction)
    bootstrapped_scores = {'NSE': [], 'KGE': [], 'r': [], 'gamma': [], 'beta': []}

    for _ in range(num_iterations):
        # Sample with replacement from the data
        sampled_data = data.sample(n=n, replace=True)
        obs_sample = sampled_data['observed'].values
        sim_sample = sampled_data['simulated'].values

        # Calculate NSE, R2, KGE for the sample
        nse_score, kge_score, r, beta, gamma = calculate_scores(obs_sample, sim_sample)

        # Store the results
        bootstrapped_scores['NSE'].append(nse_score)
        bootstrapped_scores['KGE'].append(kge_score)
        bootstrapped_scores['r'].append(r)
        bootstrapped_scores['beta'].append(beta)
        bootstrapped_scores['gamma'].append(gamma)

    return bootstrapped_scores

#--------------------------------------------------

def get_bootstrapped_scores(pth, modelname, mode):

    # mode checks
    if mode not in ["IS","OOS"]:
        raise ValueError
    if mode == "IS":
        pths = [pth+item for item in os.listdir(pth) if modelname in item and "fold" not in item]
    if mode == "OOS":
        pths = [pth+item for item in os.listdir(pth) if modelname in item and "fold" in item]

    # load all data and put into one bag
    datasets = dict()
    for i in range(len(pths)):
        with open(pths[i]+"/test_results.pickle", 'rb') as f:
            temp = pickle.load(f)
        temp = {key+"_"+str(i): value for key, value in temp.items()}
        datasets.update(temp)
    
    # get basin IDs 
    IDs = [str(item).zfill(8) for item in pd.read_csv(pth_basins, header=None).squeeze().tolist()]
    
    # container to store bootstrapped median, q05, q95
    NSE = {}
    KGE = {}
    r = {}
    gamma = {}
    beta = {}
    all_NSE = []
    
    
    # iterate over all basins to calculate bootstrapped scores
    for i in range(len(IDs)):
    
        # print progress
        if i%10==0: print(i)
        
        # get list of keys corresponding to currend ID
        IDs_i = [item for item in datasets.keys() if IDs[i] in item]
        
        # subset datasets to match only current ID
        subset_dict = {key: datasets[key] for key in IDs_i if key in datasets}
        
        # treat for the one missing time series in the OOS runs 
        if not subset_dict:
            next
        else:
            # Combine all observed and simulated data into one DataFrame
            all_data_i = pd.DataFrame(columns=['observed', 'simulated'])
            for dataset in subset_dict.items():
                df = pd.DataFrame({'observed': dataset[1]['y_obs'], 'simulated': dataset[1]['y_sim']})
                if all_data_i.empty:
                    all_data_i = copy.deepcopy(df)
                else: 
                    all_data_i = pd.concat([all_data_i, df], ignore_index=True)
            
            # Run the bootstrap procedure
            bootstrapped_i = bootstrap(all_data_i)
            all_NSE.append(bootstrapped_i["NSE"])
            
            # Calculate median and quantiles of the bootstrapped scores
            NSE_median = np.median(bootstrapped_i["NSE"])
            NSE_q05 = np.percentile(bootstrapped_i["NSE"], 5)
            NSE_q95 = np.percentile(bootstrapped_i["NSE"], 95)
            
            KGE_median = np.median(bootstrapped_i["KGE"])
            KGE_q05 = np.percentile(bootstrapped_i["KGE"], 5)
            KGE_q95 = np.percentile(bootstrapped_i["KGE"], 95)
            
            r_median = np.median(bootstrapped_i["r"])
            r_q05 = np.percentile(bootstrapped_i["r"], 5)
            r_q95 = np.percentile(bootstrapped_i["r"], 95)
            
            beta_median = np.median(bootstrapped_i["beta"])
            beta_q05 = np.percentile(bootstrapped_i["beta"], 5)
            beta_q95 = np.percentile(bootstrapped_i["beta"], 95)
            
            gamma_median = np.median(bootstrapped_i["gamma"])
            gamma_q05 = np.percentile(bootstrapped_i["gamma"], 5)
            gamma_q95 = np.percentile(bootstrapped_i["gamma"], 95)
            
            NSE.update({IDs[i]: pd.DataFrame([[IDs[i],NSE_median, NSE_q05, NSE_q95]], 
                                             columns=["gauge_id","MEDIAN", "Q05","Q95"])})
            KGE.update({IDs[i]: pd.DataFrame([[IDs[i],KGE_median, KGE_q05, KGE_q95]], 
                                             columns=["gauge_id","MEDIAN", "Q05","Q95"])})
            r.update({IDs[i]: pd.DataFrame([[IDs[i],r_median, r_q05, r_q95]], 
                                             columns=["gauge_id","MEDIAN", "Q05","Q95"])})
            beta.update({IDs[i]: pd.DataFrame([[IDs[i],beta_median, beta_q05, beta_q95]], 
                                             columns=["gauge_id","MEDIAN", "Q05","Q95"])})
            gamma.update({IDs[i]: pd.DataFrame([[IDs[i],gamma_median, gamma_q05, gamma_q95]], 
                                             columns=["gauge_id","MEDIAN", "Q05","Q95"])})
    
    NSE = pd.concat(NSE.values())
    KGE = pd.concat(KGE.values())
    r = pd.concat(r.values())
    beta = pd.concat(beta.values())
    gamma = pd.concat(gamma.values())
    NSE.set_index("gauge_id", inplace=True)
    KGE.set_index("gauge_id", inplace=True)
    r.set_index("gauge_id", inplace=True)
    beta.set_index("gauge_id", inplace=True)
    gamma.set_index("gauge_id", inplace=True)
    
    out = {"NSE": NSE,
           "KGE": KGE,
           "r": r,
           "beta":beta,
           "gamma":gamma}
     
    all_NSE = [item for sublist in all_NSE for item in sublist]

    # return(NSE, KGE, r, beta, gamma, all_NSE)
    return(out, all_NSE)

#--------------------------------------------------

def scoreplot(data, col, line="solid", alpha=1, fill=True, ax=None, medline=True):
    plotpos = np.arange(len(data)) / (len(data) - 1)
    if fill:
        plotpos_fill = pd.concat([pd.Series(plotpos), pd.Series(np.flip(plotpos))])
        ax.fill(pd.concat([data.Q95.sort_values(),
                           data.Q05.sort_values(ascending=False)]),
                plotpos_fill, color=col, alpha=alpha * 0.2)
    ax.plot(data.MEDIAN.sort_values(), plotpos, color=col, linestyle=line, alpha=alpha)
    if medline:
        ax.plot([0,np.median(data.MEDIAN)], [0.5,0.5], 
                color="grey", linestyle="solid", linewidth=0.75, alpha=alpha)
        ax.plot([np.median(data.MEDIAN),np.median(data.MEDIAN)], [0.5,-0.1],
                color=col, linestyle=line, linewidth=0.75, alpha=alpha)


#%% read scores


# bootstrap in-sample scores (and export)
scores_full_IS,     NSEall_full_IS =     get_bootstrapped_scores(pth=pth_models, modelname="full",     mode="IS")
scores_ablation_IS, NSEall_ablation_IS = get_bootstrapped_scores(pth=pth_models, modelname="ablation", mode="IS")
scores_sumstat2_IS,  NSEall_sumstat2_IS =  get_bootstrapped_scores(pth=pth_models, modelname="sumstat2",  mode="IS")
with open(pth_models+"_scores/scores_full_IS.pickle", "wb") as f: pickle.dump(scores_full_IS, f)
with open(pth_models+"_scores/scores_ablation_IS.pickle", "wb") as f: pickle.dump(scores_ablation_IS, f)
with open(pth_models+"_scores/scores_sumstat2_IS.pickle", "wb") as f: pickle.dump(scores_sumstat2_IS, f)

# bootstrap out-of-sample scores (and export)
scores_full_OOS,     NSEall_full_OOS =     get_bootstrapped_scores(pth=pth_models, modelname="full",     mode="OOS")
scores_ablation_OOS, NSEall_ablation_OOS = get_bootstrapped_scores(pth=pth_models, modelname="ablation", mode="OOS")
scores_sumstat2_OOS,  NSEall_sumstat2_OOS =  get_bootstrapped_scores(pth=pth_models, modelname="sumstat4",  mode="OOS")
with open(pth_models+"_scores/scores_full_OOS.pickle", "wb") as f: pickle.dump(scores_full_OOS, f)
with open(pth_models+"_scores/scores_ablation_OOS.pickle", "wb") as f: pickle.dump(scores_ablation_OOS, f)
with open(pth_models+"_scores/scores_sumstat2_OOS.pickle", "wb") as f: pickle.dump(scores_sumstat2_OOS, f)

# load scores
with open(pth_models+"_scores/scores_full_IS.pickle", "rb") as f: scores_full_IS = pickle.load(f)
with open(pth_models+"_scores/scores_ablation_IS.pickle", "rb") as f: scores_ablation_IS = pickle.load(f)
with open(pth_models+"_scores/scores_sumstat2_IS.pickle", "rb") as f: scores_sumstat2_IS = pickle.load(f)
with open(pth_models+"_scores/scores_full_OOS.pickle", "rb") as f: scores_full_OOS = pickle.load(f)
with open(pth_models+"_scores/scores_ablation_OOS.pickle", "rb") as f: scores_ablation_OOS = pickle.load(f)
with open(pth_models+"_scores/scores_sumstat2_OOS.pickle", "rb") as f: scores_sumstat2_OOS = pickle.load(f)

# benchmark scores
NSE_kratzert2021 = pd.read_csv(pth_benchmarks+"NSE_kratzert2021.csv", index_col = "gauge_id")
KGE_kratzert2021 = pd.read_csv(pth_benchmarks+"KGE_kratzert2021.csv", index_col = "gauge_id")


#--------------------------------------------------

# KS-test (using only the 531 NSE values for the kratzert <-> EA_camels comparison
# because that is all that is available, but using all bootstrapped NSE values for 
# the other tests because it is available and statistically more sound)
ks, pval = stats.ks_2samp(NSE_kratzert2021.MEDIAN.tolist(), 
                          scores_full_IS["NSE"].MEDIAN.tolist())
print(f"Kratzert (2021) vs. EA_camels (IS): KS Statistic={np.round(ks,3)} / P-value: {np.round(pval,3)}")

ks, pval = stats.ks_2samp(NSEall_full_IS, NSEall_sumstat2_IS)
print(f"EA_camels (IS) vs. EA_meteo (IS): KS Statistic={ks} / P-value: {pval}")

ks, pval = stats.ks_2samp(NSEall_full_OOS, NSEall_sumstat2_OOS)
print(f"EA_camels (OOS) vs. EA_meteo (OOS): KS Statistic={ks} / P-value: {pval}")


#%%  Figure 1 (NSE)


fig, axs = plt.subplots(1, 2, figsize=(12, 5))

scoreplot(data=scores_full_OOS["NSE"], col="k", line="dashed", ax=axs[0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_OOS["NSE"], col="k", line="dashed", ax=axs[0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_OOS["NSE"], col="k", line="dashed", ax=axs[0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_OOS["NSE"], col="k", line="dashed", ax=axs[0], fill=False, medline=False, alpha=0.1)

scoreplot(data=scores_full_IS["NSE"], col="tab:blue", ax=axs[0])
scoreplot(data=scores_sumstat2_IS["NSE"], col="#14a7b8", ax=axs[0])
scoreplot(data=scores_ablation_IS["NSE"], col="tab:orange", ax=axs[0])

scoreplot(data=NSE_kratzert2021, col="k", ax=axs[0], line = "dotted", fill=False)

legend_lines_0 = [Line2D([0], [0], color="k", linestyle="dotted"),
                  Line2D([0], [0], color='tab:blue'),
                  Line2D([0], [0], color='#14a7b8'),
                  Line2D([0], [0], color='tab:orange')]
legend_labels_0 = [f'$EA_{{Kratzert (2021)}}$  NSE={np.round(np.median(NSE_kratzert2021.MEDIAN),3)}',
                   f'$EA_{{CAMELS}}$         NSE={np.round(np.median(scores_full_IS["NSE"].MEDIAN),3)}',
                   f'$EA_{{meteo}}$           NSE={np.round(np.median(scores_sumstat2_IS["NSE"].MEDIAN),3)}',
                   f'$EA_{{ablated}}$          NSE={"%05.3f"%np.round(np.median(scores_ablation_IS["NSE"].MEDIAN),3)}']
axs[0].legend(legend_lines_0, legend_labels_0, loc='upper left', prop={'size': 10})

axs[0].set_xlim(0, 1)
axs[0].set_ylim(-0.025, 1.025)
axs[0].set_xlabel("NSE")
axs[0].set_ylabel("CDF")
axs[0].set_title("temporal out-of-sample / spatial in-sample (IS)",fontweight='bold')

scoreplot(data=scores_full_IS["NSE"], col="k", ax=axs[1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_IS["NSE"], col="k", ax=axs[1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_IS["NSE"], col="k", ax=axs[1], fill=False, medline=False, alpha=0.1)

scoreplot(data=scores_full_OOS["NSE"], col="tab:blue", line="dashed", ax=axs[1])
scoreplot(data=scores_sumstat2_OOS["NSE"], col="#14a7b8", line="dashed", ax=axs[1])
scoreplot(data=scores_ablation_OOS["NSE"], col="tab:orange", line="dashed", ax=axs[1])

scoreplot(data=NSE_kratzert2021, col="k", ax=axs[1], line = "dotted", fill=False, medline=False, alpha=0.1)

legend_lines_1 = [Line2D([0], [0], color='tab:blue', linestyle="dashed"),
                  Line2D([0], [0], color='#14a7b8', linestyle="dashed"),
                  Line2D([0], [0], color='tab:orange', linestyle="dashed")]
legend_labels_1 = [f'$EA_{{CAMELS}}$   NSE={"%05.3f"%np.round(np.median(scores_full_OOS["NSE"].MEDIAN),3)}',
                   f'$EA_{{meteo}}$     NSE={np.round(np.median(scores_sumstat2_OOS["NSE"].MEDIAN),3)}',
                   f'$EA_{{ablated}}$    NSE={np.round(np.median(scores_ablation_OOS["NSE"].MEDIAN),3)}']
axs[1].legend(legend_lines_1, legend_labels_1, loc='upper left', prop={'size': 10})

axs[1].set_xlim(0, 1)
axs[1].set_ylim(-0.025, 1.025)
axs[1].set_xlabel("NSE")
axs[1].set_title("temporal out-of-sample / spatial out-of-sample (OOS)",fontweight='bold')

fig.tight_layout()
fig.savefig(pth_plot+"scoreplot_NSE.pdf")


#%% Figure S1 (KGE)


fig, axs = plt.subplots(1, 2, figsize=(12, 5))

scoreplot(data=scores_full_IS["KGE"], col="tab:blue", ax=axs[0])
scoreplot(data=scores_sumstat2_IS["KGE"], col="#14a7b8", ax=axs[0])
scoreplot(data=scores_ablation_IS["KGE"], col="tab:orange", ax=axs[0])

scoreplot(data=scores_full_OOS["KGE"], col="k", line="dashed", ax=axs[0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_OOS["KGE"], col="k", line="dashed", ax=axs[0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_OOS["KGE"], col="k", line="dashed", ax=axs[0], fill=False, medline=False, alpha=0.1)

scoreplot(data=KGE_kratzert2021, col="k", ax=axs[0], line = "dotted", fill=False)

legend_lines_0 = [Line2D([0], [0], color="k", linestyle="dotted"),
                  Line2D([0], [0], color='tab:blue'),
                  Line2D([0], [0], color='#14a7b8'),
                  Line2D([0], [0], color='tab:orange')]
legend_labels_0 = [f'$EA_{{Kratzert (2021)}}$  KGE={np.round(np.median(KGE_kratzert2021.MEDIAN),3)}',
                   f'$EA_{{CAMELS}}$         KGE={"%05.3f"%np.round(np.median(scores_full_IS["KGE"].MEDIAN),3)}',
                    f'$EA_{{meteo}}$           KGE={"%05.3f"%np.round(np.median(scores_sumstat2_IS["KGE"].MEDIAN),3)}',
                    f'$EA_{{ablated}}$          KGE={"%05.3f"%np.round(np.median(scores_ablation_IS["KGE"].MEDIAN),3)}']
axs[0].legend(legend_lines_0, legend_labels_0, loc='upper left', prop={'size': 10})

axs[0].set_xlim(0, 1)
axs[0].set_ylim(-0.025, 1.025)
axs[0].set_ylabel("CDF")
axs[0].set_xlabel("KGE")
axs[0].set_title("temporal out-of-sample / spatial in-sample (IS)",fontweight='bold')

scoreplot(data=scores_full_OOS["KGE"], col="tab:blue", line="dashed", ax=axs[1])
scoreplot(data=scores_sumstat2_OOS["KGE"], col="#14a7b8", line="dashed", ax=axs[1])
scoreplot(data=scores_ablation_OOS["KGE"], col="tab:orange", line="dashed", ax=axs[1])

scoreplot(data=scores_full_IS["KGE"], col="k", ax=axs[1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_IS["KGE"], col="k", ax=axs[1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_IS["KGE"], col="k", ax=axs[1], fill=False, medline=False, alpha=0.1)

scoreplot(data=KGE_kratzert2021, col="k", ax=axs[1], line = "dotted", fill=False, medline=False, alpha=0.1)

legend_lines_1 = [Line2D([0], [0], color='tab:blue', linestyle="dashed"),
                  Line2D([0], [0], color='#14a7b8', linestyle="dashed"),
                  Line2D([0], [0], color='tab:orange', linestyle="dashed")]
legend_labels_1 = [f'$EA_{{CAMELS}}$   KGE={"%05.3f"%np.round(np.median(scores_full_OOS["KGE"].MEDIAN),3)}',
                   f'$EA_{{meteo}}$     KGE={"%05.3f"%np.round(np.median(scores_sumstat2_OOS["KGE"].MEDIAN),3)}',
                   f'$EA_{{ablated}}$    KGE={"%05.3f"%np.round(np.median(scores_ablation_OOS["KGE"].MEDIAN),3)}']
axs[1].legend(legend_lines_1, legend_labels_1, loc='upper left', prop={'size': 10})

axs[1].set_xlim(0, 1)
axs[1].set_ylim(-0.025, 1.025)
axs[1].set_xlabel("KGE")
axs[1].set_title("temporal out-of-sample / spatial out-of-sample (OOS)",fontweight='bold')

fig.tight_layout()
fig.savefig(pth_plot+"scoreplot_KGE.pdf")


#%% KGE with details (not in paper)


legend_lines_0 = [Line2D([0], [0], color='tab:blue'),
                  Line2D([0], [0], color='#14a7b8'),
                  Line2D([0], [0], color='tab:orange')]
legend_lines_1 = [Line2D([0], [0], color='tab:blue', linestyle="dashed"),
                  Line2D([0], [0], color='#14a7b8', linestyle="dashed"),
                  Line2D([0], [0], color='tab:orange', linestyle="dashed")]

#--------------------------------------------------
# KGE

fig, axs = plt.subplots(4, 2, figsize=(12, 4*4))

scoreplot(data=scores_full_IS["KGE"], col="tab:blue", ax=axs[0,0])
scoreplot(data=scores_sumstat2_IS["KGE"], col="#14a7b8", ax=axs[0,0])
scoreplot(data=scores_ablation_IS["KGE"], col="tab:orange", ax=axs[0,0])

scoreplot(data=scores_full_OOS["KGE"], col="k", line="dashed", ax=axs[0,0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_OOS["KGE"], col="k", line="dashed", ax=axs[0,0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_OOS["KGE"], col="k", line="dashed", ax=axs[0,0], fill=False, medline=False, alpha=0.1)

legend_labels_0 = [f'$EA_{{CAMELS}}$   KGE={"%05.3f"%np.round(np.median(scores_full_IS["KGE"].MEDIAN),3)}',
                    f'$EA_{{meteo}}$     KGE={"%05.3f"%np.round(np.median(scores_sumstat2_IS["KGE"].MEDIAN),3)}',
                    f'$EA_{{ablated}}$    KGE={"%05.3f"%np.round(np.median(scores_ablation_IS["KGE"].MEDIAN),3)}']
axs[0,0].legend(legend_lines_0, legend_labels_0, loc='upper left', prop={'size': 10})

axs[0,0].set_xlim(0, 1)
axs[0,0].set_ylabel("CDF")
axs[0,0].set_title("temporal out-of-sample / spatial in-sample (IS)\n\n KGE",fontweight='bold')

scoreplot(data=scores_full_OOS["KGE"], col="tab:blue", line="dashed", ax=axs[0,1])
scoreplot(data=scores_sumstat2_OOS["KGE"], col="#14a7b8", line="dashed", ax=axs[0,1])
scoreplot(data=scores_ablation_OOS["KGE"], col="tab:orange", line="dashed", ax=axs[0,1])

scoreplot(data=scores_full_IS["KGE"], col="k", ax=axs[0,1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_IS["KGE"], col="k", ax=axs[0,1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_IS["KGE"], col="k", ax=axs[0,1], fill=False, medline=False, alpha=0.1)

legend_labels_1 = [f'$EA_{{CAMELS}}$   KGE={"%05.3f"%np.round(np.median(scores_full_OOS["KGE"].MEDIAN),3)}',
                   f'$EA_{{meteo}}$     KGE={"%05.3f"%np.round(np.median(scores_sumstat2_OOS["KGE"].MEDIAN),3)}',
                   f'$EA_{{ablated}}$    KGE={"%05.3f"%np.round(np.median(scores_ablation_OOS["KGE"].MEDIAN),3)}']
axs[0,1].legend(legend_lines_1, legend_labels_1, loc='upper left', prop={'size': 10})

axs[0,1].set_xlim(0, 1)
axs[0,1].set_title("temporal out-of-sample / spatial out-of-sample (OOS)\n\n KGE",fontweight='bold')

#--------------------------------------------------
# r

scoreplot(data=scores_full_IS["r"], col="tab:blue", ax=axs[1,0])
scoreplot(data=scores_sumstat2_IS["r"], col="#14a7b8", ax=axs[1,0])
scoreplot(data=scores_ablation_IS["r"], col="tab:orange", ax=axs[1,0])

scoreplot(data=scores_full_OOS["r"], col="k", line="dashed", ax=axs[1,0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_OOS["r"], col="k", line="dashed", ax=axs[1,0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_OOS["r"], col="k", line="dashed", ax=axs[1,0], fill=False, medline=False, alpha=0.1)

legend_labels_0 = [f'$EA_{{CAMELS}}$   r={"%05.3f"%np.round(np.median(scores_full_IS["r"].MEDIAN),3)}',
                    f'$EA_{{meteo}}$     r={"%05.3f"%np.round(np.median(scores_sumstat2_IS["r"].MEDIAN),3)}',
                    f'$EA_{{ablated}}$    r={"%05.3f"%np.round(np.median(scores_ablation_IS["r"].MEDIAN),3)}']
axs[1,0].legend(legend_lines_0, legend_labels_0, loc='upper left', prop={'size': 10})

axs[1,0].set_xlim(0, 1)
axs[1,0].set_title("r (correlation)",fontweight='bold')
axs[1,0].set_ylabel("CDF")

scoreplot(data=scores_full_OOS["r"], col="tab:blue", line="dashed", ax=axs[1,1])
scoreplot(data=scores_sumstat2_OOS["r"], col="#14a7b8", line="dashed", ax=axs[1,1])
scoreplot(data=scores_ablation_OOS["r"], col="tab:orange", line="dashed", ax=axs[1,1])

scoreplot(data=scores_full_IS["r"], col="k", ax=axs[1,1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_IS["r"], col="k", ax=axs[1,1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_IS["r"], col="k", ax=axs[1,1], fill=False, medline=False, alpha=0.1)

legend_labels_1 = [f'$EA_{{CAMELS}}$   r={"%05.3f"%np.round(np.median(scores_full_OOS["r"].MEDIAN),3)}',
                   f'$EA_{{meteo}}$     r={"%05.3f"%np.round(np.median(scores_sumstat2_OOS["r"].MEDIAN),3)}',
                   f'$EA_{{ablated}}$    r={"%05.3f"%np.round(np.median(scores_ablation_OOS["r"].MEDIAN),3)}']
axs[1,1].legend(legend_lines_1, legend_labels_1, loc='upper left', prop={'size': 10})

axs[1,1].set_xlim(0, 1)
axs[1,1].set_title("r (correlation)",fontweight='bold')

#--------------------------------------------------
# beta 

scoreplot(data=scores_full_IS["beta"], col="tab:blue", ax=axs[2,0])
scoreplot(data=scores_sumstat2_IS["beta"], col="#14a7b8", ax=axs[2,0])
scoreplot(data=scores_ablation_IS["beta"], col="tab:orange", ax=axs[2,0])

scoreplot(data=scores_full_OOS["beta"], col="k", line="dashed", ax=axs[2,0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_OOS["beta"], col="k", line="dashed", ax=axs[2,0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_OOS["beta"], col="k", line="dashed", ax=axs[2,0], fill=False, medline=False, alpha=0.1)

legend_labels_0 = [f'$EA_{{CAMELS}}$   beta={"%05.3f"%np.round(np.median(scores_full_IS["beta"].MEDIAN),3)}',
                    f'$EA_{{meteo}}$     beta={"%05.3f"%np.round(np.median(scores_sumstat2_IS["beta"].MEDIAN),3)}',
                    f'$EA_{{ablated}}$    beta={"%05.3f"%np.round(np.median(scores_ablation_IS["beta"].MEDIAN),3)}']
axs[2,0].legend(legend_lines_0, legend_labels_0, loc='upper left', prop={'size': 10})

axs[2,0].set_xlim(0, 1.7)
axs[2,0].set_ylabel("CDF")
axs[2,0].set_title("beta (mean bias)",fontweight='bold')

scoreplot(data=scores_full_OOS["beta"], col="tab:blue", line="dashed", ax=axs[2,1])
scoreplot(data=scores_sumstat2_OOS["beta"], col="#14a7b8", line="dashed", ax=axs[2,1])
scoreplot(data=scores_ablation_OOS["beta"], col="tab:orange", line="dashed", ax=axs[2,1])

scoreplot(data=scores_full_IS["beta"], col="k", ax=axs[2,1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_IS["beta"], col="k", ax=axs[2,1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_IS["beta"], col="k", ax=axs[2,1], fill=False, medline=False, alpha=0.1)

legend_labels_1 = [f'$EA_{{CAMELS}}$   beta={"%05.3f"%np.round(np.median(scores_full_OOS["beta"].MEDIAN),3)}',
                   f'$EA_{{meteo}}$     beta={"%05.3f"%np.round(np.median(scores_sumstat2_OOS["beta"].MEDIAN),3)}',
                   f'$EA_{{ablated}}$    beta={"%05.3f"%np.round(np.median(scores_ablation_OOS["beta"].MEDIAN),3)}']
axs[2,1].legend(legend_lines_1, legend_labels_1, loc='upper left', prop={'size': 10})

axs[2,1].set_xlim(0, 1.7)
axs[2,1].set_title("beta (mean bias)",fontweight='bold')

#--------------------------------------------------
# gamma

scoreplot(data=scores_full_IS["gamma"], col="tab:blue", ax=axs[3,0])
scoreplot(data=scores_sumstat2_IS["gamma"], col="#14a7b8", ax=axs[3,0])
scoreplot(data=scores_ablation_IS["gamma"], col="tab:orange", ax=axs[3,0])

scoreplot(data=scores_full_OOS["gamma"], col="k", line="dashed", ax=axs[3,0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_OOS["gamma"], col="k", line="dashed", ax=axs[3,0], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_OOS["gamma"], col="k", line="dashed", ax=axs[3,0], fill=False, medline=False, alpha=0.1)

legend_labels_0 = [f'$EA_{{CAMELS}}$   gamma={"%05.3f"%np.round(np.median(scores_full_IS["gamma"].MEDIAN),3)}',
                    f'$EA_{{meteo}}$     gamma={"%05.3f"%np.round(np.median(scores_sumstat2_IS["gamma"].MEDIAN),3)}',
                    f'$EA_{{ablated}}$    gamma={"%05.3f"%np.round(np.median(scores_ablation_IS["gamma"].MEDIAN),3)}']
axs[3,0].legend(legend_lines_0, legend_labels_0, loc='upper left', prop={'size': 10})

axs[3,0].set_xlim(0, 1.7)
axs[3,0].set_ylabel("CDF")
axs[3,0].set_title("gamma (variability bias)",fontweight='bold')

scoreplot(data=scores_full_OOS["gamma"], col="tab:blue", line="dashed", ax=axs[3,1])
scoreplot(data=scores_sumstat2_OOS["gamma"], col="#14a7b8", line="dashed", ax=axs[3,1])
scoreplot(data=scores_ablation_OOS["gamma"], col="tab:orange", line="dashed", ax=axs[3,1])

scoreplot(data=scores_full_IS["gamma"], col="k", ax=axs[3,1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_sumstat2_IS["gamma"], col="k", ax=axs[3,1], fill=False, medline=False, alpha=0.1)
scoreplot(data=scores_ablation_IS["gamma"], col="k", ax=axs[3,1], fill=False, medline=False, alpha=0.1)

legend_labels_1 = [f'$EA_{{CAMELS}}$   gamma={"%05.3f"%np.round(np.median(scores_full_OOS["gamma"].MEDIAN),3)}',
                   f'$EA_{{meteo}}$     gamma={"%05.3f"%np.round(np.median(scores_sumstat2_OOS["gamma"].MEDIAN),3)}',
                   f'$EA_{{ablated}}$    gamma={"%05.3f"%np.round(np.median(scores_ablation_OOS["gamma"].MEDIAN),3)}']
axs[3,1].legend(legend_lines_1, legend_labels_1, loc='upper left', prop={'size': 10})

axs[3,1].set_xlim(0, 1.7)
axs[3,1].set_title("gamma (variability bias)",fontweight='bold')

#--------------------------------------------------

fig.tight_layout()
fig.savefig(pth_plot+"scoreplot_KGE_long.pdf")



#%% Figure S3 (map)


diff_IS = (pd.merge(scores_full_IS["NSE"].rename(columns={"MEDIAN":"full_IS"}).full_IS, 
                   scores_ablation_IS["NSE"].rename(columns={"MEDIAN":"ablation_IS"}).ablation_IS,
                   left_index=True, right_index=True)
           .assign(diff_IS=lambda x: x["ablation_IS"]-x["full_IS"])
           .drop(["full_IS","ablation_IS"], axis=1)
           .assign(diff_IS=lambda x: x["diff_IS"].clip(lower=-4.99))
           .assign(diff_IS=lambda x: x["diff_IS"].clip(upper=0.99)))

diff_OOS = (pd.merge(scores_full_OOS["NSE"].rename(columns={"MEDIAN":"full_OOS"}).full_OOS, 
                    scores_ablation_OOS["NSE"].rename(columns={"MEDIAN":"ablation_OOS"}).ablation_OOS,
                    left_index=True, right_index=True)
            .assign(diff_OOS=lambda x: x["ablation_OOS"]-x["full_OOS"])
            .drop(["full_OOS","ablation_OOS"], axis=1)
            .assign(diff_OOS=lambda x: x["diff_OOS"].clip(lower=-4.99))
            .assign(diff_OOS=lambda x: x["diff_OOS"].clip(upper=0.99)))

diff_OOS_meteo = (pd.merge(scores_full_OOS["NSE"].rename(columns={"MEDIAN":"full_OOS"}).full_OOS, 
                          scores_sumstat2_OOS["NSE"].rename(columns={"MEDIAN":"sumstat2_OOS"}).sumstat2_OOS,
                          left_index=True, right_index=True)
                  .assign(diff_OOS_meteo=lambda x: x["sumstat2_OOS"]-x["full_OOS"])
                  .drop(["full_OOS","sumstat2_OOS"], axis=1)
                  .assign(diff_OOS_meteo=lambda x: x["diff_OOS_meteo"].clip(lower=-4.99))
                  .assign(diff_OOS_meteo=lambda x: x["diff_OOS_meteo"].clip(upper=0.99)))


# load coordinates, subset to selection, strip unused columns
coords = pd.read_csv(pth_coords, delimiter=";")
basins = pd.read_csv(pth_basins, header=None).squeeze().tolist()
coords_where = [index for index, item in enumerate(coords["gauge_id"]) if item in basins]
coords["gauge_id"] = [str(item).zfill(8) for item in coords["gauge_id"]]
coords = coords.iloc[coords_where,:3].set_index("gauge_id")

# add data to plot
coords = pd.merge(pd.merge(pd.merge(coords, diff_IS, left_index=True, right_index=True),
                           diff_OOS, left_index=True, right_index=True),
                  diff_OOS_meteo, left_index=True, right_index=True)

# extract lat/lon
lat = coords["gauge_lat"]
lon = coords["gauge_lon"]

# Make  gdfs of coords and usa-map
gdf_coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
gdf_shape = gpd.read_file(pth_usashape)


# create discrete classes for plotting
gdf_coords['class_diff_IS'] = pd.cut(gdf_coords['diff_IS'], 
                              bins=[-5,-1,-0.5,-0.25,0,0.25,0.5,1])

gdf_coords['class_diff_OOS'] = pd.cut(gdf_coords['diff_OOS'], 
                              bins=[-5,-1,-0.5,-0.25,0,0.25,0.5,1])

gdf_coords['class_diff_OOS_meteo'] = pd.cut(gdf_coords['diff_OOS_meteo'], 
                                            bins=[-5,-1,-0.5,-0.25,0,0.25,0.5,1])

# plot
fig, axs = plt.subplots(3, 1, figsize=(12, 4*3))
fig.suptitle("drop in NSE", fontweight="bold")

# create color map
RdBu = plt.colormaps.get_cmap('RdBu')
colors = [RdBu(0.05), RdBu(0.15), RdBu(0.25), RdBu(0.35),
          RdBu(0.6), RdBu(0.75), RdBu(0.9)]
cmap = ListedColormap(colors)

axs[0].set_title("$EA_{{CAMELS}} \Longrightarrow EA_{{ablated}}$ (IS)")
gdf_shape.plot(ax=axs[0], color='none', edgecolor='grey', alpha=0.7)
gdf_coords.plot(ax=axs[0], column='class_diff_IS', cmap=cmap)
axs[0].set_xlim([-126, -61])  
axs[0].set_ylim([24, 50]) 
axs[0].get_xaxis().set_visible(False)

axs[1].set_title("$EA_{{CAMELS}} \Longrightarrow EA_{{ablated}}$ (OOS)")
gdf_shape.plot(ax=axs[1], color='none', edgecolor='grey', alpha=0.7)
gdf_coords.plot(ax=axs[1], column='class_diff_OOS', cmap=cmap, legend=True)
axs[1].set_xlim([-126, -61])  
axs[1].set_ylim([24, 50]) 
axs[1].get_xaxis().set_visible(False)

axs[2].set_title("$EA_{{CAMELS}} \Longrightarrow EA_{{meteo}}$ (OOS)")
gdf_shape.plot(ax=axs[2], color='none', edgecolor='grey', alpha=0.7)
gdf_coords.plot(ax=axs[2], column='class_diff_OOS_meteo', cmap=cmap)
axs[2].set_xlim([-126, -61])  
axs[2].set_ylim([24, 50]) 

fig.tight_layout()
fig.savefig(pth_plot+"map_NSE.pdf")
