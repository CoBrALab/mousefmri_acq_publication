#!/usr/bin/env python
# coding: utf-8

# run in py38 env
# # This script is added to address reviewer comments. It computes avg phgy data within 2-min windows WITHOUT phgy censoring.
# # this will allow me to look at relationship between phgy and fmri censoring since the two are not removed simultaneously
##PROBLEM NEED CPP dataset_zscore and sklearn csvs
import dual_regression_functions
import numpy as np
import glob
import pandas as pd
import os
import pickle
import sys

#extract sub_ses_id
sub_ses_ID=str(sys.argv[1])
sub = str(sys.argv[2])
ses = str(sys.argv[3])
scan_num=int(sys.argv[4])

#load data
final_data_folder='/data/chamal/projects/mila/2021_fMRI_dev/part2_phgy_fmri_project/4_derivatives/rabies_runs/local_data_final_runs'
spatial_prior_IC_files = "../../1_reference_data/local_data/melodic_IC_resampled_to_local_data.nii.gz"
dataset_info_df = pd.read_csv("../../2_raw_data/dataset_specs.csv").set_index("epi_filename")
epi_files_local = sorted(glob.glob(final_data_folder + '/rabies_out_cc-v050_05smoothed_lowpass/confound_correction_datasink/cleaned_timeseries/*/*rest*'))

#load the phgy derivatives and FD data too
final_phgy_data_folder='../../4_derivatives/phgy_derivatives/physiology_analysis_outputs'
rabies_FD_csv = sorted(glob.glob(final_data_folder + '/rabies_out_preprocess-v050/motion_datasink/FD_csv/*/*/' + (sub_ses_ID) + '*rest*.csv'))[0]
resp_csv = sorted(glob.glob(final_phgy_data_folder + '/dx' + ses + '_' + sub + '*resp*window.csv'))[0]
pulseox_csv = sorted(glob.glob(final_phgy_data_folder + '/dx' + ses + '_' + sub + '*pleth*window.csv'))[0]
spo2_csv = sorted(glob.glob('../../3_preprocessed_data/phgy_data_oldNaming/dx' + ses + '_' + sub + '*spo2*'))[0]
CPP_datasetzscore_df = pd.read_csv(os.path.abspath('../CPP_analysis/intermediary_outputs_for_sanity_check/CPP_metrics_5clust_dataset_zscore.csv'))
CPP_sexstrainzscore_df = pd.read_csv(os.path.abspath('../CPP_analysis/final_output/CPP_metrics_5clust_sexstrain_zscore.csv'))
CPP_sexstrainzscore_sklearn_df = pd.read_csv(os.path.abspath('../CPP_analysis/intermediary_outputs_for_sanity_check/CPP_metrics_5clust_sexstrain_zscore_sklearn.csv'))

#censoring dict that contains fmri, phgy and nan censoring
fmri_nan_censoring_dict = pickle.load(open('../../4_derivatives/final_phgy_censored/pickle_files/dict_fmri_nan_censor', "rb"))

#extract details about the specfic session from the dataset_info_df and find the right files
session_info  = dataset_info_df.loc[sub_ses_ID, :]
session_info_columns = list(dataset_info_df.columns)
columns_to_copy = [0,1,2,3,4,5,6,8]
epi_file = epi_files_local[scan_num]

resp_df = pd.read_csv(resp_csv).drop(columns = ['Unnamed: 0', 'Window start time','Window end time'], axis = 1)
pulseox_df = pd.read_csv(pulseox_csv).drop(columns = ['Unnamed: 0', 'Window start time','Window end time'], axis = 1)
spo2_df = pd.read_csv(spo2_csv, names = ['SpO2 (%)'])
rabies_FD_df = pd.read_csv(os.path.abspath(rabies_FD_csv))

indices_CPP_dataset_thisScan= np.where(CPP_datasetzscore_df['index'] == sub_ses_ID)[0]
CPP_datasetzscore_thisScan_df = CPP_datasetzscore_df.loc[indices_CPP_dataset_thisScan][['corr_to_clust0', 'corr_to_clust1', 'corr_to_clust2', 'corr_to_clust3', 'corr_to_clust4', 'cluster_prediction']]
indices_CPP_sexstrain_thisScan= np.where(CPP_sexstrainzscore_df['index'] == sub_ses_ID)[0]
CPP_sexstrainzscore_thisScan_df = CPP_sexstrainzscore_df.loc[indices_CPP_sexstrain_thisScan][['corr_to_clust0', 'corr_to_clust1', 'corr_to_clust2', 'corr_to_clust3', 'corr_to_clust4', 'cluster_prediction' ,'RR', 'RRV', 'RV', 'HR', 'HRV', 'PVI', 'SpO2']]
CPP_sexstrainzscore_sklearn_thisScan_df = CPP_sexstrainzscore_sklearn_df.loc[indices_CPP_sexstrain_thisScan][['dist_to_clust0', 'dist_to_clust1', 'dist_to_clust2', 'dist_to_clust3', 'dist_to_clust4', 'cluster_prediction']]
#convert csv to df
censoring_df_1440length= fmri_nan_censoring_dict[sub_ses_ID].to_frame(name = "False = Masked Frames").reset_index()

#print status
print('Processing file: ' + str(os.path.basename(epi_file)) + ' out of ' + str(len(epi_files_local)) + ' files, of length: ' + str(np.sum(censoring_df_1440length)[1]))
print('Resp file is: ' + str(os.path.basename(resp_csv)))
print('Pulseox file is: ' + str(os.path.basename(pulseox_csv)))
print('SpO2 file is: ' + str(os.path.basename(spo2_csv)))
print('FD file is: ' + str(os.path.basename(rabies_FD_csv)))
print('CPP datasetzscore data has length ' + str(len(indices_CPP_dataset_thisScan)))
print('CPP sexstrain zscore data has length ' + str(len(indices_CPP_sexstrain_thisScan)))

#calculate the phgy metrics in the same time window
dual_regression_functions.extract_phgy_in_window(resp_df, pulseox_df, spo2_df, rabies_FD_df, CPP_datasetzscore_thisScan_df, CPP_sexstrainzscore_thisScan_df, CPP_sexstrainzscore_sklearn_thisScan_df, censoring_df_1440length,
                                                    1440, 120, 0,  session_info,
                                                    "./intermediary_outputs_sanity_check/phgy_windowAvg_outputs_nophgycensored/" + sub_ses_ID, True)