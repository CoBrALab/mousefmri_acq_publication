#!/usr/bin/env python
# coding: utf-8

# # Run in py38 environment

''' The purpose is to combine the physiology metrics and DR regression outputs within 2 min time windows.
This csv of all the variables combined is used later in statistical analyses. 

Additionally, we investigate the effect of various choices on the metrics derived from DR.
For example, how the outputs compare if we replace the censored timepoints with the mean of the timeseries.
How Dice, correlation and fit variance relate, and how the somatomotor and dmn networks differ.
'''


import numpy as np
import glob
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 
plt.rc('axes', labelsize=15) 

#load all the csvs
phgy_files = sorted(glob.glob("./intermediary_outputs_sanity_check/phgy_windowAvg_outputs/*.csv"))
epi_dr_window_csvs = sorted(glob.glob("./intermediary_outputs_sanity_check/DR_outputs/*.csv"))
epi_dr_window_variableLength_csvs = sorted(glob.glob("./intermediary_outputs_sanity_check/DR_outputs_variableWindows/*.csv"))
dataset_specs_df = pd.read_csv('../../2_raw_data/dataset_specs.csv', usecols = ['epi_filename', 'subject_ID', 'session_ID',
                                                                             'dex_val', 'dex_level', 'dex_conc', 
                                                                             'actual_ses_order', 'strain', 'sex', 'ID',
                                                                             'weight', 'age_days'])

#check that there are no files missing
print('Num phgy csvs: ' + str(len(phgy_files)))
print('Num DR csvs (censored to mean): ' + str(len(epi_dr_window_csvs)))
print('Num DR csvs (variable windows): ' + str(len(epi_dr_window_variableLength_csvs)))

def combine_dr_phgy_csvs(phgy_csvs, dr_csvs, dataset_specs, output_name):

    #combine the csvs of each type to form one master csv
    df_per_sub_dict = {}
    arr_all_subs = np.zeros((0,114))
    num_sessions = 46
    for file in range (0,num_sessions):
        #extract name
        file_name = os.path.basename(phgy_csvs[file])
        sub = file_name[7:10]
        ses = file_name[15:16]

        #locate appropriate csvs for that sub
        dr_df = pd.read_csv(dr_csvs[file]).reset_index(drop = True)
        phgy_df = pd.read_csv(phgy_csvs[file]).reset_index(drop = True)
        
        #now also add the dataset_info
        session_info = pd.DataFrame(np.repeat(dataset_specs.loc[(dataset_specs['epi_filename'] =='sub-PHG' + sub + '_ses-' + ses)].values, 12, axis=0), columns = dataset_specs.columns)

        #combine dfs into one per subject
        df_per_sub_final = pd.concat([session_info,
                                    phgy_df.drop(['Unnamed: 0'], axis = 1), 
                                    dr_df.drop(['Unnamed: 0', 'Start Time Realtime', 'End Time Realtime',
                                                    'Start Time Censoredtime', 'End Time Censoredtime',
                                                    'Number of Timepoints', 'Iso percent', 'Iso val', 'Dice DMN neg', 'Dice DMN comb'], axis = 1)],
                                    axis=1)
        #rename the DMN dice column to avoid problems
        df_per_sub_final = df_per_sub_final.rename(columns = {'Dice DMN pos': 'Dice DMN'})
        #convert the dataframe to list (for easier appending)
        df_per_sub_dict[sub + '_' + ses] = df_per_sub_final
        
        #append to existing list 
        arr_all_subs = np.concatenate((arr_all_subs, np.array(df_per_sub_final)), axis=0)

    #convert final list to dataframe
    df_all_subs = pd.DataFrame(arr_all_subs, columns =df_per_sub_final.columns)
    df_all_subs.to_csv(output_name + '.csv')
    print('Original number of timepoints: '  + str(df_all_subs.shape))

    #drop rows containing nans or too few timepoints (<70)
    df_all_subs = df_all_subs.dropna(axis = 0)
    print('Number of timepoints after dropping nans: ' + str(df_all_subs.shape))
    df_all_subs = df_all_subs[df_all_subs['Number of Timepoints'] > 80]
    print('Number of timepoints fter dropping windows with <80 timepoints: ' + str(df_all_subs.shape))

    df_all_subs.to_csv(output_name + '_sparse.csv')
    return df_all_subs

df_all_subs_censoredtomean = combine_dr_phgy_csvs(phgy_files, epi_dr_window_csvs, dataset_specs_df, './master_DR_and_phgy_window')
df_all_subs_variableLength = combine_dr_phgy_csvs(phgy_files, epi_dr_window_variableLength_csvs, dataset_specs_df, './master_DR_and_phgy_variableWindow')

#####################################plot the relationship between different types of DR metrics ####################################
fig, axs = plt.subplots(3, 3, figsize=(16, 16), sharey = True)
dict_networks = {0: 'Somatomotor', 1: 'Sensorivisual', 2: 'DMN'}
for i in dict_networks:
    axs[i,0].plot(df_all_subs_censoredtomean['Correlation ' + str(dict_networks[i])], df_all_subs_censoredtomean['Dice ' + str(dict_networks[i])], '.')
    axs[i,0].set_title('Pearson correlation between metrics: ' + str(round(stats.pearsonr(df_all_subs_censoredtomean['Dice ' + str(dict_networks[i])], df_all_subs_censoredtomean['Correlation ' + str(dict_networks[i])])[0], 2)))
    axs[i,1].plot(df_all_subs_censoredtomean['Covariance ' + str(dict_networks[i])],df_all_subs_censoredtomean['Dice ' + str(dict_networks[i])], '.')
    axs[i,1].set_title('Pearson correlation between metrics: ' + str(round(stats.pearsonr(df_all_subs_censoredtomean['Dice ' + str(dict_networks[i])], df_all_subs_censoredtomean['Covariance ' + str(dict_networks[i])])[0], 2)))
    axs[i,2].plot(df_all_subs_censoredtomean['Fit Variance ' + str(dict_networks[i])], df_all_subs_censoredtomean['Dice ' + str(dict_networks[i])], '.')
    axs[i,2].set_title('Pearson correlation between metrics: ' + str(round(stats.pearsonr(df_all_subs_censoredtomean['Dice ' + str(dict_networks[i])], df_all_subs_censoredtomean['Fit Variance ' +  str(dict_networks[i])])[0], 2)), fontsize = 15)
    #set axis labels
    axs[i,0].set_xlabel('Correlation ' + str(dict_networks[i]))
    axs[i,1].set_xlabel('Covariance ' + str(dict_networks[i]))
    axs[i,2].set_xlabel('Fit Variance ' + str(dict_networks[i]))
    axs[i,0].set_ylabel('Dice ' + str(dict_networks[i]))

    #plot diagonal line
    line_arr = np.arange(0,1, 0.01)
    axs[i, 0].plot(line_arr, line_arr)
    axs[i, 1].plot(line_arr, line_arr)
    axs[i, 2].plot(line_arr, line_arr)
fig.savefig('./final_outputs/figures/comparison_DR_metrics.svg')

############################ plot the correlation between the version where censored windows are replaced with the window mean VS when the windows have a variable number of timepoint

line_arr = np.arange(0,1, 0.01)

fig2, axs = plt.subplots(2, 3, figsize = (19,14), sharey = False)
dict_networks = {0: 'Somatomotor', 1: 'Sensorivisual', 2: 'DMN'}
for i in dict_networks:
    axs[0,i].scatter(df_all_subs_censoredtomean['Dice ' + str(dict_networks[i])],
                 df_all_subs_variableLength['Dice ' + str(dict_networks[i])], c = df_all_subs_variableLength['Number of Timepoints']  )
    axs[0,i].set_xlabel('Dice (censored to mean)')
    axs[0,i].set_ylabel('Dice (variable window lengths)')
    axs[0,i].plot(line_arr, line_arr)
    axs[0,i].set_title('Network ' + str(dict_networks[i]) + ', \n metric correlation = ' + str(round(stats.pearsonr(df_all_subs_censoredtomean['Dice ' + str(dict_networks[i])], df_all_subs_variableLength['Dice ' + str(dict_networks[i])])[0], 4)), fontsize = 15)
    axs[0,i].set_xlim([0,1])
    axs[0,i].set_ylim([0,1])
    
    axs[1,i].scatter(df_all_subs_censoredtomean['Correlation ' + str(dict_networks[i])],
                 df_all_subs_variableLength['Correlation ' + str(dict_networks[i])], c = df_all_subs_variableLength['Number of Timepoints'] )
    axs[1,i].set_xlabel('Correlation (censored to mean)')
    axs[1,i].set_ylabel('Correlation (variable window lengths)')
    axs[1,i].set_title('Network ' + str(dict_networks[i]) + ', \n metric correlation = ' + str(round(stats.pearsonr(df_all_subs_censoredtomean['Correlation ' + str(dict_networks[i])], df_all_subs_variableLength['Correlation ' + str(dict_networks[i])])[0], 4)), fontsize = 15)
    axs[1,i].plot(line_arr, line_arr)
    axs[1,i].set_xlim([0,1])
    axs[1,i].set_ylim([0,1])

    print('Network ' + str(dict_networks[i]) + ', metric correlation = ' + str(round(stats.pearsonr(df_all_subs_censoredtomean['Dice ' + str(dict_networks[i])], df_all_subs_censoredtomean['Dice ' + str(dict_networks[i])])[0], 4)))
    print('Network ' + str(dict_networks[i]) + ', metric correlation = ' + str(round(stats.pearsonr(df_all_subs_censoredtomean['Correlation ' + str(dict_networks[i])], df_all_subs_censoredtomean['Correlation ' + str(dict_networks[i])])[0], 4)))
fig2.savefig("./final_outputs/figures/comparison_DR_censoring_approach.svg")

######################### plot the histogram of correlation values ##########################################

#also include the CAP histogram for comparison
cap_df = pd.read_csv('/data/chamal/projects/mila/2021_fMRI_dev/part2_phgy_fmri_project/5_analysis/CPP_analysis/CAP_analysis/CAP_metrics_KMeansCorr.csv')
sns.set(style="ticks", font_scale = 3)
fig3, axs = plt.subplots(4, 1, figsize=(12, 32))
dict_networks = {0: 'Correlation Somatomotor', 1: 'Correlation DMN', 2: 'Average correlation to network'}
xlabel_names = ['Spatial Correlation to Somatomotor Network', 'Spatial Correlation to DMN-like Network', 'Average Spatial Correlation' ]
network_colors = {0: 'skyblue', 1: 'deepskyblue', 2: 'gray'}
for i in dict_networks:
    sns.histplot(data=df_all_subs_variableLength, x=str(dict_networks[i]), kde=True, color=network_colors[i], ax=axs[i], bins = 50)
    axs[i].set_ylabel('Number of 2-min windows')
    axs[i].set_xlabel(str(xlabel_names[i]))

#now plot the cap distribution separately
sns.histplot(data=np.abs(cap_df), x='Corr-networkAvg', kde=True, color='gray', ax=axs[i+1], bins = 50)
axs[i+1].set_ylabel('Number of timepoints')
axs[i+1].set_xlabel('CAP - average correlation to network')
fig3.savefig('./final_outputs/figures/DR_correlation_histograms.svg')
######################### relationship between somatomotor and dmn ##########################################

fig4, axs = plt.subplots(1, 1, figsize=(10, 10))
myplot = plt.scatter(df_all_subs_variableLength['Correlation Somatomotor'],
                 df_all_subs_variableLength['Correlation DMN'], c = df_all_subs_variableLength['Average correlation to network'] )
plt.xlabel('Somatomotor correlation')
plt.ylabel('DMN correlation')
cbar = plt.colorbar(myplot)

fig4.savefig('./final_outputs/figures/DR_somat-dmn_corr.png')

