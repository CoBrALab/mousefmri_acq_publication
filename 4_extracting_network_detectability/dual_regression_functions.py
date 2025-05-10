#!/usr/bin/env python
# coding: utf-8

# # Goal: Fit spatial priors to individual subs using dual-regression and Gabe's special new function in a time window

# In[1]:


from sklearn.utils import check_random_state
import numpy as np
import nilearn
import nibabel as nb
import rabies.preprocess_pkg.utils
from nilearn.plotting import plot_stat_map
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os

# import functions for dual ICA from rabies
from rabies.analysis_pkg.data_diagnosis import resample_IC_file
from rabies.analysis_pkg.analysis_functions import closed_form
from rabies.analysis_pkg.prior_modeling import _logcosh


# In[2]:


#redefine deflation_fit here to include a seed (the original RABIES func does not use a seed)
def deflation_fit(X, q=1, c_init=None, C_convergence='OLS', C_prior=None, W_prior=None, W_ortho=False, tol=1e-6, 
                  max_iter=200, verbose=1):
    # q defines the number of new sources to fit
    if c_init is None:
        random_state = check_random_state(0)
        c_init = random_state.normal(
            size=(X.shape[1], q))

    # the C_prior and W_prior correspond to spatial and temporal priors respectively which will impose an orthogonality contraint
    # on the fitted sources in their respective dimension
    if C_prior is None:
        C_prior = np.zeros([X.shape[1], 0])
    C_prior /= np.sqrt((C_prior ** 2).sum(axis=0))

    if W_prior is None:
        W_prior = np.zeros([X.shape[0], 0])
    W_prior /= np.sqrt((W_prior ** 2).sum(axis=0))

    # initialize an empty C
    C = np.zeros([X.shape[1], 0])
    for j in range(q):
        C_prev = np.concatenate((C, C_prior), axis=1)
        c = c_init[:, j].reshape(-1, 1)
        c /= np.sqrt((c ** 2).sum(axis=0))

        # regress out the orthogonal dimensions already fitted
        X_ = X-np.matmul(np.matmul(X, C_prev), C_prev.T)

        for i in range(max_iter):
            c_prev = c

            w = np.matmul(X_, c)
            if W_ortho and (W_prior.shape[1]>0):
                # impose complete temporal orthogonality, more similar to CR before but not quite the same
                w -= np.matmul(W_prior, closed_form(W_prior, w))
            W = np.concatenate((w,W_prior),axis=1) # include the W priors in the convergence step

            if C_convergence == 'OLS':
                c = closed_form(W, X_).T[:,0].reshape(-1, 1) # take back only c
            elif C_convergence == 'ICA':
                gwtx, g_wtx = _logcosh(W[:,0].reshape(1,-1), {})
                c = ((X_.T * gwtx).mean(axis=1) - g_wtx.mean() * c.T).T
            else:
                raise

            # impose spatial orthogonality
            c -= np.matmul(np.matmul(c.T, C_prev), C_prev.T).T
            c /= np.sqrt((c ** 2).sum(axis=0))

            ##### evaluate convergence
            lim = np.abs(np.abs((c * c_prev).sum(axis=0)) - 1).mean()
            if verbose > 2:
                print('lim:'+str(lim))
            if lim < tol:
                if verbose > 1:
                    print(str(i)+' iterations to converge.')
                break
            if i == max_iter-1:
                if verbose > 0:
                    print(
                        'Convergence failed. Consider increasing max_iter or decreasing tol.')
        C = np.concatenate((C, c), axis=1)
    return C


# In[3]:


def dual_regression(all_IC_vectors, timeseries):
    ### compute dual regression
    ### Here, we adopt an approach where the algorithm should explain the data
    ### as a linear combination of spatial maps. The data itself, is only temporally
    ### detrended, and not spatially centered, which could cause inconsistencies during
    ### linear regression according to https://mandymejia.com/2018/03/29/the-role-of-centering-in-dual-regression/#:~:text=Dual%20regression%20requires%20centering%20across%20time%20and%20space&text=time%20points.,each%20time%20course%20at%20zero
    ### The fMRI timeseries aren't assumed theoretically to be spatially centered, and
    ### this measure would be removing global signal variations which we are interested in.
    ### Thus we prefer to avoid this step here, despite modelling limitations.
    X = all_IC_vectors.T
    Y = timeseries.T
    # for one given volume, it's values can be expressed through a linear combination of the components
    W = closed_form(X, Y, intercept=False).T
    # normalize the component timecourses to unit variance
    W /= W.std(axis=0)
    # for a given voxel timeseries, it's signal can be explained a linear combination of the component timecourses
    C = closed_form(W, Y.T, intercept=False)
    DR = {'C':C, 'W':W}
    return DR


# In[51]:


#the fitting function must be redefined here since I changed deflation_fit function
def dual_ICA_fit(timeseries, num_comp, all_IC_vectors, prior_bold_idx):
    prior_fit_out={'C':[],'W':[]}
    convergence_function = 'ICA'
    X=timeseries

    prior_networks = all_IC_vectors[prior_bold_idx,:].T

    C_prior=prior_networks
    C_conf = deflation_fit(X, q=num_comp, c_init=None, C_convergence=convergence_function,
                      C_prior=C_prior, W_prior=None, W_ortho=True, tol=1e-6, max_iter=200, verbose=1)
    for network in range(prior_networks.shape[1]):
        prior=prior_networks[:,network].reshape(-1,1)
        C_prior=np.concatenate((prior_networks[:,:network],prior_networks[:,network+1:],C_conf),axis=1)

        C_fit = deflation_fit(X, q=1, c_init=prior, C_convergence=convergence_function,
                              C_prior=C_prior, W_prior=None, W_ortho=True, tol=1e-6, max_iter=200, verbose=1)

        # make sure the sign of weights is the same as the prior
        corr = np.corrcoef(C_fit.flatten(), prior.flatten())[0, 1]
        if corr < 0:
            C_fit = C_fit*-1

        # the finalized C
        C = np.concatenate((C_fit, C_prior), axis=1)

        # L-2 norm normalization of the components
        C /= np.sqrt((C ** 2).sum(axis=0))
        W = closed_form(C,X.T, intercept=False).T
        # the components will contain the weighting/STD/singular value, and the timecourses are normalized
        C=C*W.std(axis=0)
        # normalize the component timecourses to unit variance
        W /= W.std(axis=0)

        prior_fit_out['C'].append(C[:,0])
        prior_fit_out['W'].append(W[:,0])
    return prior_fit_out


# In[4]:


#define function for plotting
def recover_3D(mask_file, vector_map):
    brain_mask = np.asarray(nb.load(mask_file).dataobj)
    volume_indices = brain_mask.astype(bool)
    volume = np.zeros(brain_mask.shape)
    volume[volume_indices] = vector_map
    volume_img = nb.Nifti1Image(volume, nb.load(mask_file).affine, nb.load(mask_file).header)
    return volume_img


# In[5]:


#define function that extracts vector from a image file
def preprocess_nifti(nifti_path_string, mask_path_string, smooth_bool, smooth_level):
    
    #load the image as an array
    nb_obj = nb.load(nifti_path_string)
    nifti = np.asarray(nb_obj.dataobj)
    mask= np.asarray(nb.load(mask_path_string).dataobj)
    
    #optional - smooth the nifti
    if smooth_bool:
        nifti_smoothed = np.asarray(nilearn.image.smooth_img(nb_obj, smooth_level).dataobj)
        nifti = nifti_smoothed
        
    #apply mask to epi (this also flattens it)
    volume_indices = mask.astype(bool)
    nifti_masked = nifti[volume_indices]
        
    return nifti, mask, nifti_masked

def dice_coefficient(mask1,mask2):
    #this is from rabies/analysis_pkg/analysis_math
    dice = np.sum(mask1*mask2)*2.0 / (np.sum(mask1) + np.sum(mask2))
    return dice

def replace_censored_with_mean(epi, censored_timepoint_df, num_timepoints):
    '''This function takes the input censored timeseries (either 4D or flattened 2D) and inserts the mean images
    across time at the timepoints that have been censored. The output will thus have same number of timepoints as the 
    original (uncensored) EPI.'''
    
    #find indices of timepoints that were NOT censored
    indices_uncensored = np.where(censored_timepoint_df == True)[0]
    
    #determine whether the input is 4D or 2D
    epi_dim = len(epi.shape)
    
    #take the mean image across all timepoints
    mean_image = np.mean(epi, axis = epi_dim - 1)
    
    if epi_dim == 4:
        epi_original_length = np.repeat(mean_image[:,:,:,np.newaxis], num_timepoints, axis = 3)
        epi_original_length[:,:,:,indices_uncensored] = epi
    elif epi_dim ==2:
        epi_original_length = np.repeat(mean_image[:,np.newaxis], num_timepoints, axis = 1)
        epi_original_length[:,indices_uncensored] = epi
    else:
        print('Improper input EPI dimensions')
    return epi_original_length

def percent_masking(array, percentile_top, include_neg, percentile_bottom):
    threshold_lower = None
    #this is modified from rabies/analysis_pkg/diagnosis_pkg/diagnosis_QC
    flat=array.flatten()
    sorted_ascending = np.sort(flat)
    
    # To extract top positive voxels
    idx_top=int((1-percentile_top)*len(flat)) #out of the total number of voxels, find how many are in the top 99%
    threshold_top = sorted_ascending[idx_top] #extract the value of the voxel at the 99% position (since array is sorted)
    mask_top = array >= threshold_top #find voxels where the original array is above the 99% threshold
    
    if include_neg:
        #To extract top most negative voxels.
        idx_bottom=int((1-percentile_bottom)*len(flat))
        sorted_descending = sorted_ascending[::-1]
        threshold_bottom = sorted_descending[idx_bottom]
        mask_bottom=array<=threshold_bottom

        #combine the two masks into one
        mask_comb = mask_top + mask_bottom
        return mask_top, mask_bottom, mask_comb
        
    else:
        return mask_top, None, None
    
def get_df_avg_over_overlapping_windows(df_original, window_start_realtime,window_overlap, num_timepoints, session_info, 
                                       session_info_columns,fit_metrics, df_fit_metrics, columns_to_copy):
        ############################################## AVERAGE OVER OVERLAPPING WINDOWS #####################
        #now average the correlations from two overlapping time windows to get values for every 30 s
        df_avg_info = pd.DataFrame()
        df_avg_info['Start Time'] = np.append(df_original['Start Time Realtime'], window_start_realtime)
        df_avg_info['End Time'] = np.append(df_original['End Time Realtime']-window_overlap, num_timepoints)
        df_avg_info['Iso percent'] = np.append(df_original['Iso percent'], session_info['iso_percent_high'])
        df_avg_info['Iso val'] = np.append(df_original['Iso val'], session_info['iso_level_high'])
        df_avg_fit = pd.DataFrame((np.pad(fit_metrics,((0,1), (0,0)), 'edge') + np.pad(fit_metrics,((1,0), (0,0)),'edge'))/2, 
                                 columns = df_fit_metrics.columns)
        ############################################ Add session info ######################################
        for column_num in columns_to_copy:
            df_avg_info[session_info_columns[column_num]] = np.repeat(session_info[column_num], len(df_avg_info))
        df_avg = pd.concat([df_avg_info, df_avg_fit], axis = 1)
    
        return df_avg
    
def extract_indiv_fit_metrics(desired_component_to_fit, epi_segment, window_start_realtime, mask_file, background_im, 
                              prior_mask_pos, prior_mask_neg, prior_mask_comb, output_name, output_name_nifti, idx, fit_type, DR_fit_plot_vmax):
    ################################################# FITTING ##################################################
    fit_out_C = None
    try:
        if fit_type == 'DR':
            desired_component_to_fit_reshaped = np.reshape(desired_component_to_fit, (1,-1))
            fit_out = dual_regression(desired_component_to_fit_reshaped, epi_segment)
            fit_out_C = fit_out['C'][0]
            vmax_fit = DR_fit_plot_vmax
        else:
            #perform gabe's new spatiotemporal fitting
            #Fit 10 spatial and 10 temporal confound components to account for remaining variance
            desired_component_to_fit_reshaped = np.reshape(desired_component_to_fit, (-1,1))
            #fit_out = spatiotemporal_prior_fit_functions.spatiotemporal_prior_fit(epi_segment,
            #                                                                      desired_component_to_fit_reshaped, 10,10)
            fit_out_C = np.squeeze(fit_out['C_fitted_prior'])
            vmax_fit = 6*DR_fit_plot_vmax
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            print('window starting at ' + str(window_start_realtime) + ' had singular matrix - ' + fit_type)
        else:
            raise
        return [-1, -1, -1, -1, -1, -1]
            
    #smooth the output
    volume_indices=np.asarray(nb.load(mask_file).dataobj).astype(bool) 
    fit_out_smoothed = np.array(nilearn.image.smooth_img(recover_3D(mask_file, 
                                                                    fit_out_C), 0.3).dataobj)[volume_indices]
    #recover fit output as 3d array
    fit_out_3d_img = recover_3D(mask_file, fit_out_smoothed)
    fit_out_3d_arr = fit_out_3d_img.get_fdata()

    #save the output as a nifti
    nb.save(fit_out_3d_img, output_name_nifti + '_fit_' + fit_type + '_prior' + str(idx) + "_time_" + str(window_start_realtime) + ".nii.gz")
    ############################################## EXTRACT METRICS #############################################
    #calculate the correlation of the subject-specific fitted component with the original group-level component
    fit_to_prior_corr = stats.pearsonr(fit_out_smoothed, desired_component_to_fit)

    #calculate the covariance as well
    fit_to_prior_cov = np.cov(fit_out_smoothed, desired_component_to_fit)
    
    #calculate the standard deviation of the fitted time series
    fit_var_exp = np.std(fit_out_C)

    #calculate the dice overlap of thresholded masks from fit_out and spatial_prior
    fit_mask_pos, _, _ = percent_masking(fit_out_3d_arr, 0.01, False, 0)
    dice_coeff_pos = dice_coefficient(fit_mask_pos, prior_mask_pos)
    nb.save(nb.Nifti1Image(fit_mask_pos, nb.load(mask_file).affine,nb.load(mask_file).header), output_name_nifti + '_fit_' + fit_type + '_prior' + str(idx) + "_time_" + str(window_start_realtime) + "_mask.nii.gz")

    ########################################### Account for positive and negative ###########################

    #if multiple prior masks are provided (eg for the top negative voxels too), also calc those dice overlaps
    if (prior_mask_neg is None):
        #plot the tresholded fit masks in each time window
        fig1,axs = plt.subplots(nrows=2, ncols=1, figsize=(12,6))
        axs[0].set_title(fit_type + ' Fitted prior - Window start:' + str(window_start_realtime))
        axs[1].set_title('Top positive voxels mask')
        plot_stat_map(fit_out_3d_img,bg_img=background_im, axes = axs[0], cut_coords=(0,1,2,3,4,5),
                  display_mode='y', colorbar=True, vmax = vmax_fit)
        plot_stat_map(nb.Nifti1Image(fit_mask_pos, nb.load(mask_file).affine,nb.load(mask_file).header),
                      bg_img=background_im, axes = axs[1], cut_coords=(0,1,2,3,4,5), display_mode='y', colorbar=True)
        plt.savefig(output_name + '_fit_' + fit_type + '_prior' + str(idx) + "_time_" + str(window_start_realtime) + ".png")
        plt.close()
        return [fit_to_prior_corr[0], fit_to_prior_cov[1,0], fit_var_exp, dice_coeff_pos]

    else:
        #the negative 
        fit_mask_pos, fit_mask_neg, fit_mask_comb = percent_masking(fit_out_3d_arr, 0.01, True, 0.005)
        dice_coeff_neg = dice_coefficient(fit_mask_neg, prior_mask_neg)
        dice_coeff_comb = dice_coefficient(fit_mask_comb, prior_mask_comb)

        #plot the other thresholded fit masks too
        fig2,axs = plt.subplots(nrows=4, ncols=1, figsize=(12,12))
        axs[0].set_title(fit_type + ' Fitted prior - Window start:' + str(window_start_realtime))
        axs[1].set_title('Top positive voxels mask')
        axs[2].set_title('Top negative voxels mask')
        axs[3].set_title('Combined mask')
        plot_stat_map(fit_out_3d_img,bg_img=background_im, axes = axs[0], cut_coords=(0,1,2,3,4,5),
                  display_mode='y', colorbar=True, vmax = vmax_fit)
        plot_stat_map(nb.Nifti1Image(fit_mask_pos, nb.load(mask_file).affine,nb.load(mask_file).header),
                      bg_img=background_im, axes = axs[1], cut_coords=(0,1,2,3,4,5), display_mode='y', colorbar=True)
        plot_stat_map(nb.Nifti1Image(fit_mask_neg, nb.load(mask_file).affine,nb.load(mask_file).header),
                      bg_img=background_im, axes = axs[2], cut_coords=(0,1,2,3,4,5), display_mode='y', colorbar=True)
        plot_stat_map(nb.Nifti1Image(fit_mask_comb, nb.load(mask_file).affine,nb.load(mask_file).header),
                      bg_img=background_im, axes = axs[3], cut_coords=(0,1,2,3,4,5), display_mode='y', colorbar=True)
        plt.savefig(output_name+ '_fit_' + fit_type + '_prior' + str(idx) + "_time_" + str(window_start_realtime) + ".png")
        plt.close()
        return [fit_to_prior_corr[0], fit_to_prior_cov[1,0], fit_var_exp, dice_coeff_pos, dice_coeff_neg, dice_coeff_comb]

def perform_indiv_fit_window_multiprior(epi_file, censoring_df_1440length, censoring_df_afterFmriCensoringLength, mask_file, background_im, priors_file, 
                                  prior_indices, prior_percent_masking_param, num_timepoints, window_width, window_overlap, 
                                  session_info, session_info_columns, columns_to_copy, DR_fit_plot_vmax, output_name, output_name_nifti, replace_censored_with_mean_bool):
    """This version accounts for censored timepoints and overlapping windows - FINAL VERSION"""
    #process the EPI and spatial priors files
    _,_, epi_masked_flat_fmriCensOnly = preprocess_nifti(epi_file,mask_file, False, 0.0)
    priors_arr_smooth, _, priors_arr_masked_smooth = preprocess_nifti(priors_file,mask_file, True, 0.5)

    #censor the EPI according to phgy too
    print('Original dimensions: ' + str(epi_masked_flat_fmriCensOnly.shape))
    epi_masked_flat = pd.DataFrame(epi_masked_flat_fmriCensOnly).loc[:, censoring_df_afterFmriCensoringLength['False = Masked Frames']].reset_index(drop = True).to_numpy()
    print('Confirmation that file ' + str(os.path.basename(epi_file)) + ' has size' + str(epi_masked_flat.shape))

    #create arrays to store the correlation values for for all windows
    if window_overlap > 0:
        num_windows = 1+int((num_timepoints - window_width)/window_overlap) #numerator gives last start time, frac gives num starts
    else:
        num_windows = int(num_timepoints/window_width)
    window_time_info = np.zeros((num_windows,7))
    DR_fit_metrics = np.zeros((num_windows, 14))
    iso_info = np.zeros((num_windows,2))
    
    ################################################### Loop over time windows ####################################
    #extract a time window
    window_start_realtime = 0 
    window_start_censoredtime = 0 
    window_count = 0
    while window_start_realtime + window_width <= num_timepoints:
        
        #calculate when the time window should end (according to the length of the original, uncensored data)
        window_end_realtime = window_start_realtime + window_width
        
        #calculate how many timepoints actually remain in the window after censoring (by counting # of TRUE entries)
        censored_in_window = censoring_df_1440length['False = Masked Frames'][window_start_realtime:window_end_realtime]

        try:
            num_uncensored_timepoints = censored_in_window.value_counts()[1] 
            window_end_censoredtime = window_start_censoredtime + num_uncensored_timepoints
            
            #extract only the part of the EPI inside the time window (it must be transposed to put time dimension first)
            epi_segment = epi_masked_flat[:, window_start_censoredtime:window_end_censoredtime]
            
            #restore the length of the timeseries within the window by replacing censored points with mean (in window)
            #then transpose it to put time dim first
            if replace_censored_with_mean_bool == True:
                epi_segment = np.transpose(replace_censored_with_mean(epi_segment,censored_in_window, window_width))
                num_timepoints_for_DR = epi_segment.shape[0]
            else:
                epi_segment = np.transpose(epi_segment)
                num_timepoints_for_DR = epi_segment.shape[0]
        
            ############################################ Prior fit - loop over priors ########################################
            i = 0
            DR_fit_metrics_all_priors = []
            
            #perform the dual-regression fitting for each spatial prior
            for idx in prior_indices:
                #extract the prior masks (top x % of voxels) according to provided params
                prior_mask_pos, prior_mask_neg, prior_mask_comb = percent_masking(priors_arr_smooth[:,:,:,idx], 
                                                                                *prior_percent_masking_param[i,:])
                
                #perform the fit, plot results and extract metrics
                DR_fit_metrics_per_prior = extract_indiv_fit_metrics(priors_arr_masked_smooth[:,idx],epi_segment,
                                                                    window_start_realtime, mask_file,background_im, 
                                                                    prior_mask_pos,prior_mask_neg, prior_mask_comb,output_name, output_name_nifti,
                                                                    idx, 'DR', DR_fit_plot_vmax)
                DR_fit_metrics_all_priors.extend(DR_fit_metrics_per_prior)
                i=i+1
        except:
            num_uncensored_timepoints = 0
            window_end_censoredtime = window_start_censoredtime + num_uncensored_timepoints
            DR_fit_metrics_all_priors = [np.nan]*14
            num_timepoints_for_DR = 0

        ############################################ Out of priors loop ######################################
        #store information from this window in arrays
        time_since_iso_change = window_start_realtime % 480
        window_time_info[window_count, :] = [window_start_realtime, window_end_realtime, window_start_censoredtime, 
                                             window_end_censoredtime, num_uncensored_timepoints, num_timepoints_for_DR, time_since_iso_change]
        DR_fit_metrics[window_count, :] = DR_fit_metrics_all_priors
        
        #add iso information from this window
        if window_start_realtime < 480:
            iso_info[window_count,:] = [session_info['iso_percent_low'], session_info['iso_level_low']] 
        elif (window_start_realtime >= 480) & (window_start_realtime <960):
            iso_info[window_count,:] = [session_info['iso_percent_mid'], session_info['iso_level_mid']] 
        elif window_start_realtime >= 960:
            iso_info[window_count,:] = [session_info['iso_percent_high'], session_info['iso_level_high']] 
                
        #set the start time of the next window in realtime
        diff_end_overlap = window_end_realtime - window_overlap
        window_start_realtime = diff_end_overlap
        
        #calculate how many timepoints are in the overlap after censoring, set start time in censoredtime
        if window_overlap >0:
            num_uncensored_timepoints_overlap = censoring_df_1440length[diff_end_overlap:window_end_realtime].value_counts()[1]
        else:
            num_uncensored_timepoints_overlap = 0
        window_start_censoredtime = window_end_censoredtime - num_uncensored_timepoints_overlap
        window_count = window_count + 1
        
    ################################################## Out of windows loop #######################################    
    #now add the information to a dataframe
    df_time = pd.DataFrame(window_time_info, columns=['Start Time Realtime', 'End Time Realtime', 'Start Time Censoredtime',
                                                     'End Time Censoredtime', 'Number of Timepoints', 
                                                     'Number of Timepoints in DR window (inc mean)', 'Time after isoflurane change'])
    df_iso = pd.DataFrame(iso_info, columns = ['Iso percent', 'Iso val'])
    df_DRfit_metrics = pd.DataFrame(DR_fit_metrics, columns = ['Correlation Somatomotor', 'Covariance Somatomotor', 'Fit Variance Somatomotor',
                                                          'Dice Somatomotor','Correlation Sensorivisual', 'Covariance Sensorivisual', 
                                                          'Fit Variance Sensorivisual', 'Dice Sensorivisual','Correlation DMN', 
                                                          'Covariance DMN', 'Fit Variance DMN','Dice DMN pos', 
                                                          'Dice DMN neg', 'Dice DMN comb'])
    df_original_DR = pd.concat([df_time, df_iso, df_DRfit_metrics], axis = 1)

    #also add a column that combines sOMATOMOTOR and DMN correlations
    df_original_DR['Average correlation to network'] = (np.abs(df_original_DR['Correlation Somatomotor']) + np.abs(df_original_DR['Correlation DMN']))/2
    
    ################################################## Average over overlapping windows ########################
    if window_overlap > 0:
        df_avg_DR = get_df_avg_over_overlapping_windows(df_original_DR, window_start_realtime,window_overlap, num_timepoints, 
                                                        session_info, session_info_columns,DR_fit_metrics, df_DRfit_metrics, columns_to_copy, output_name_nifti,)
        df_avg_DR.to_csv(output_name + "_overlap_avg_DR.csv")
    #save the dataframes as a csv
    df_original_DR.to_csv(output_name + "_fit_DR.csv")
    print('Finished running: ' + output_name)
 
def perform_indiv_fit_window_multiprior_figurever(epi_file, censored_timepoint_df, mask_file, background_im, priors_file, 
                                  prior_indices, prior_percent_masking_param, num_timepoints, window_width, window_overlap, 
                                  session_info, session_info_columns, columns_to_copy, output_name):
    """This version accounts for censored timepoints and overlapping windows - this version is for plotting posters figures
    Instead of plotting the DR fit and mask separately, it overlaps the two."""
    #process the EPI and spatial priors files
    _,_, epi_masked_flat = preprocess_nifti(epi_file,mask_file, False, 0.3)
    priors_arr_smooth, _, priors_arr_masked_smooth = preprocess_nifti(priors_file,mask_file, True, 0.4)

    #create arrays to store the correlation values for for all windows
    num_windows = 1+int((num_timepoints - window_width)/window_overlap) #numerator gives last start time, frac gives num starts
    window_time_info = np.zeros((num_windows,5))
    DR_fit_metrics = np.zeros((num_windows, 14))
    SPT_fit_metrics = np.zeros((num_windows, 14))
    iso_info = np.zeros((num_windows,2))
    
    ################################################### Loop over time windows ####################################
    #extract a time window
    window_start_realtime = 0 
    window_start_censoredtime = 0 
    window_count = 0
    while window_start_realtime + window_width <= num_timepoints:
        
        #calculate when the time window should end (according to the length of the original, uncensored data)
        window_end_realtime = window_start_realtime + window_width
        
        #calculate how many timepoints actually remain in the window after censoring (by counting # of TRUE entries)
        censored_in_window = censored_timepoint_df['False = Masked Frames'][window_start_realtime:window_end_realtime]
        num_uncensored_timepoints = censored_in_window.value_counts()[1]
        window_end_censoredtime = window_start_censoredtime + num_uncensored_timepoints
        
        #extract only the part of the EPI inside the time window (it must be transposed to put time dimension first)
        epi_segment = epi_masked_flat[:, window_start_censoredtime:window_end_censoredtime]
        
        #restore the length of the timeseries within the window by replacing censored points with mean (in window)
        #then transpose it to put time dim first
        epi_segment_full_length = np.transpose(replace_censored_with_mean(epi_segment,censored_in_window, window_width))
    
        ############################################ Prior fit - loop over priors ########################################
        i = 0
        DR_fit_metrics_all_priors = []
        spt_fit_metrics_all_priors = []
        
        #perform the dual-regression fitting and gabe's new spatiotemporal (spt) fit algorithm - for each spatial prior
        for idx in prior_indices:
            #extract the prior masks (top x % of voxels) according to provided params
            prior_mask_pos, prior_mask_neg, prior_mask_comb = percent_masking(priors_arr_smooth[:,:,:,idx], 
                                                                              *prior_percent_masking_param[i,:])
            
            #perform the fit, plot results and extract metrics
            DR_fit_metrics_per_prior = extract_indiv_fit_metrics(priors_arr_masked_smooth[:,idx],epi_segment_full_length,
                                                                 window_start_realtime, mask_file,background_im, 
                                                                 prior_mask_pos,prior_mask_neg, prior_mask_comb,output_name, output_name_nifti, 
                                                                 idx, 'DR')
            spt_fit_metrics_per_prior = extract_indiv_fit_metrics(priors_arr_masked_smooth[:,idx],epi_segment_full_length,
                                                                  window_start_realtime, mask_file,background_im, 
                                                                  prior_mask_pos,prior_mask_neg, prior_mask_comb,output_name, output_name_nifti,
                                                                  idx, 'spt')
            DR_fit_metrics_all_priors.extend(DR_fit_metrics_per_prior)
            spt_fit_metrics_all_priors.extend(spt_fit_metrics_per_prior)
            i=i+1

        ############################################ Out of priors loop ######################################
        #store information from this window in arrays
        window_time_info[window_count, :] = [window_start_realtime, window_end_realtime, window_start_censoredtime, 
                                             window_end_censoredtime, num_uncensored_timepoints]
        DR_fit_metrics[window_count, :] = DR_fit_metrics_all_priors
        SPT_fit_metrics[window_count, :] = spt_fit_metrics_all_priors
        
        #add iso information from this window
        if window_start_realtime < 480:
            iso_info[window_count,:] = [session_info['iso_percent_low'], session_info['iso_level_low']] 
        elif (window_start_realtime >= 480) & (window_start_realtime <960):
            iso_info[window_count,:] = [session_info['iso_percent_mid'], session_info['iso_level_mid']] 
        elif window_start_realtime >= 960:
            iso_info[window_count,:] = [session_info['iso_percent_high'], session_info['iso_level_high']] 
                
        #set the start time of the next window in realtime
        diff_end_overlap = window_end_realtime - window_overlap
        window_start_realtime = diff_end_overlap
        
        #calculate how many timepoints are in the overlap after censoring, set start time in censoredtime
        num_uncensored_timepoints_overlap = censored_timepoint_df[diff_end_overlap:window_end_realtime].value_counts()[1]
        window_start_censoredtime = window_end_censoredtime - num_uncensored_timepoints_overlap
        window_count = window_count + 1
        
    ################################################## Out of windows loop #######################################    
    #now add the information to a dataframe
    df_time = pd.DataFrame(window_time_info, columns=['Start Time Realtime', 'End Time Realtime', 'Start Time Censoredtime',
                                                     'End Time Censoredtime', 'Number of Timepoints'])
    df_iso = pd.DataFrame(iso_info, columns = ['Iso percent', 'Iso val'])
    df_DRfit_metrics = pd.DataFrame(DR_fit_metrics, columns = ['Correlation Somatomotor', 'Covariance Somatomotor', 'Fit Variance Somatomotor',
                                                          'Dice Somatomotor','Correlation Sensorivisual', 'Covariance Sensorivisual', 
                                                          'Fit Variance Sensorivisual', 'Dice Sensorivisual','Correlation DMN', 
                                                          'Covariance DMN', 'Fit Variance DMN','Dice DMN pos', 
                                                          'Dice DMN neg', 'Dice DMN comb'])
    df_SPTfit_metrics = pd.DataFrame(SPT_fit_metrics, columns = ['Correlation Somatomotor', 'Covariance Somatomotor', 'Fit Variance Somatomotor',
                                                          'Dice Somatomotor','Correlation Sensorivisual', 'Covariance Sensorivisual', 
                                                          'Fit Variance Sensorivisual', 'Dice Sensorivisual','Correlation Sensorivisual', 
                                                          'Covariance DMN', 'Fit Variance DMN','Dice DMN pos', 
                                                          'Dice DMN neg', 'Dice DMN comb'])
    df_original_DR = pd.concat([df_time, df_iso, df_DRfit_metrics], axis = 1)
    df_original_SPT = pd.concat([df_time, df_iso, df_SPTfit_metrics], axis = 1)
    
    ################################################## Average over overlapping windows ########################
    df_avg_DR = get_df_avg_over_overlapping_windows(df_original_DR, window_start_realtime,window_overlap, num_timepoints, 
                                                    session_info, session_info_columns,DR_fit_metrics, df_DRfit_metrics, columns_to_copy)
    df_avg_SPT = get_df_avg_over_overlapping_windows(df_original_SPT, window_start_realtime,window_overlap, num_timepoints, 
                                                    session_info, session_info_columns,SPT_fit_metrics, df_SPTfit_metrics, columns_to_copy)
    
    #save the dataframes as a csv
    df_original_DR.to_csv(output_name + "_fit_DR.csv")
    df_original_SPT.to_csv(output_name + "_fit_SPT.csv")
    df_avg_DR.to_csv(output_name + "_overlap_avg_DR.csv")
    df_avg_SPT.to_csv(output_name + "_overlap_avg_SPT.csv")
    print('Finished running: ' + output_name)
    
    return 
    
def extract_phgy_in_window(resp_df, pulseox_df, spo2_df, rabies_FD_df, CPP_datasetzscore_df, CPP_sexstrainzscore_df, CPP_sexstrainzscore_sklearn_df, censoring_df_1440length,
                           num_timepoints, window_width, window_overlap,  session_info,
                           output_name, replace_censored_with_mean_bool):
    '''This function combines the resp and pulseox csvs for a given subject/session, drops unimportant variables
    and censors according to the RABIES temporal mask. Finds mean in window.'''
    #concatenate resp, pulseox, spo2 and FD together
    phgy_df_allvar = pd.concat([resp_df.add_suffix('-resp'), pulseox_df.add_suffix('-pleth'), spo2_df, rabies_FD_df.add_suffix(' FD')]
                               , axis = 1)
    CPP_df = pd.concat([CPP_datasetzscore_df.add_suffix('-dataset_zscore'), CPP_sexstrainzscore_df.add_suffix('-sexstrain_zscore'), CPP_sexstrainzscore_sklearn_df.add_suffix('-sexstrain_zscore_sklearn')], axis = 1)
    CPP_prediction_df = CPP_df[['cluster_prediction-dataset_zscore', 'cluster_prediction-sexstrain_zscore', 'cluster_prediction-sexstrain_zscore_sklearn']]
    CPP_df = CPP_df.drop(['cluster_prediction-dataset_zscore', 'cluster_prediction-sexstrain_zscore', 'cluster_prediction-sexstrain_zscore_sklearn'], axis = 1)

    #create arrays to store the correlation values for for all windows
    if window_overlap > 0:
        num_windows = 1+int((num_timepoints - window_width)/window_overlap) #numerator gives last start time, frac gives num starts
    else:
        num_windows = int(num_timepoints/window_width)

    
        #drop redundant variables that are highly correlated (see next section for the justification)
    #also drop std in window, entropy and pulse width because they're harder to interpret and less grounded in lit
    phgy_df_selectvar = phgy_df_allvar.drop(columns = ['Period-overall window mean-resp',
                                                        'Resp rate-overall window-resp',
                                                        'Instantanous RRV period rmssd-window mean-resp',
                                                        'RRV-overall period window rmssd-resp',
                                                        'RRV-overall period window std-resp',
                                                        'Period-overall window mean-pleth',
                                                        'HR-overall window-pleth',
                                                        'Instantanous HRV period rmssd-window mean-pleth',
                                                        'HRV-overall period window rmssd-pleth',
                                                        'HRV-overall period window std-pleth',
                                                        'Instantaneous resp rate - window std-resp',
                                                        'Instantaneous HR - window std-pleth',
                                                        'Instantaneous entropy-window mean-resp',
                                                        'Instantaneous entropy-window mean-pleth',
                                                        'Width at quarter height-overall window mean-pleth',
                                                        'width at half height-overall window mean-pleth',
                                                        'Width at base-overall window mean-pleth'], axis = 1)
    #rename the existing column names to simpler ones
    phgy_df_selectvar = phgy_df_selectvar.rename(columns={"Instantaneous resp rate-window mean-resp": "RR",
                                                            "Instantaneous RV - window mean-resp": "RV", 
                                                            "Instantaneous RRV period std-window mean-resp": "RRV",
                                                            "Instantaneous HR-window mean-pleth": "HR",
                                                            "Instantaneous HRV period std-window mean-pleth": "HRV",
                                                            "Instantaneous PVI-window mean-pleth": "PVI", 
                                                            "SpO2 (%)": "SpO2"})
    #change order of RV and RRV (to better parallel the order of pulse ox metrics)
    phgy_df_selectvar.insert(1, 'RRV', phgy_df_selectvar.pop('RRV'))
    phgy_df_selectvar = phgy_df_selectvar.astype(float)

    window_time_info = np.zeros((num_windows,5))
    phgy_metrics_mean = np.zeros((num_windows, len(phgy_df_selectvar.columns)))
    phgy_metrics_std = np.zeros((num_windows, len(phgy_df_selectvar.columns)))
    CPP_metrics_mean = np.zeros((num_windows, 22))
    CPP_metrics_std = np.zeros((num_windows, 22))
    cluster_prediction_counts = np.zeros((num_windows, 15))
    cluster_prediction_most_freq = np.zeros((num_windows, 3))
    iso_info = np.zeros((num_windows,2))

    #################################################### Apply censoring ###############################3
    phgy_df_selectvar = phgy_df_selectvar.loc[censoring_df_1440length['False = Masked Frames'], :].reset_index(drop = True)

    ################################################### Loop over time windows ####################################
    #extract a time window
    window_start_realtime = 0 
    window_start_censoredtime = 0 
    window_count = 0
    while window_start_realtime + window_width <= num_timepoints:
        
        #calculate when the time window should end (according to the length of the original, uncensored data)
        window_end_realtime = window_start_realtime + window_width
        
        #calculate how many timepoints actually remain in the window after censoring (by counting # of TRUE entries)
        censored_in_window = censoring_df_1440length['False = Masked Frames'][window_start_realtime:window_end_realtime]
        try:
            num_uncensored_timepoints = censored_in_window.value_counts()[1] 
        except:
            num_uncensored_timepoints = 0
        window_end_censoredtime = window_start_censoredtime + num_uncensored_timepoints
        
        #extract only the part inside the time window
        phgy_df_selectvar_segment = phgy_df_selectvar[window_start_censoredtime:window_end_censoredtime]
        cpp_segment = CPP_df[window_start_censoredtime:window_end_censoredtime]
        cluster_prediction_segment = CPP_prediction_df[window_start_censoredtime:window_end_censoredtime]
        
        ##################### Extract mean in window ###################
        phgy_metrics_mean_in_window = np.mean(phgy_df_selectvar_segment, axis = 0)
        phgy_metrics_std_in_window = np.std(phgy_df_selectvar_segment, axis = 0)
        CPP_metrics_mean_in_window = np.mean(cpp_segment, axis = 0)
        CPP_metrics_std_in_window = np.std(cpp_segment, axis = 0)
        
        #store information from this window in arrays
        window_time_info[window_count, :] = [window_start_realtime, window_end_realtime, window_start_censoredtime, 
                                             window_end_censoredtime, num_uncensored_timepoints]
        phgy_metrics_mean[window_count, :] = phgy_metrics_mean_in_window
        phgy_metrics_std[window_count, :] = phgy_metrics_std_in_window
        CPP_metrics_mean[window_count, :] = CPP_metrics_mean_in_window
        CPP_metrics_std[window_count, :] = CPP_metrics_std_in_window

        #for the cluster prediction information, count the number of timepoints in each cluster df.loc[df['column_name'] == some_value]
        cluster_counts = [len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-dataset_zscore'] == 0]), 
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-dataset_zscore'] == 1]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-dataset_zscore'] == 2]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-dataset_zscore'] == 3]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-dataset_zscore'] == 4]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-sexstrain_zscore'] == 0]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-sexstrain_zscore'] == 1]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-sexstrain_zscore'] == 2]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-sexstrain_zscore'] == 3]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-sexstrain_zscore'] == 4]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-sexstrain_zscore_sklearn'] == 0]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-sexstrain_zscore_sklearn'] == 1]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-sexstrain_zscore_sklearn'] == 2]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-sexstrain_zscore_sklearn'] == 3]),
                            len(cluster_prediction_segment.loc[cluster_prediction_segment['cluster_prediction-sexstrain_zscore_sklearn'] == 4])]
        try:
            cluster_counts_normalized = [x / num_uncensored_timepoints for x in cluster_counts]
            cluster_prediction_counts[window_count, :] = cluster_counts_normalized
            cluster_prediction_most_freq[window_count, :] = [cluster_prediction_segment['cluster_prediction-dataset_zscore'].mode()[0], cluster_prediction_segment['cluster_prediction-sexstrain_zscore'].mode()[0],
                                                               cluster_prediction_segment['cluster_prediction-sexstrain_zscore_sklearn'].mode()[0]] 

        except:
            cluster_counts_normalized = cluster_counts #in the case that num_uncensored timepoints is 0, don't bother dividing
            cluster_prediction_most_freq[window_count, :] = [np.nan, np.nan, np.nan]
        #add iso information from this window
        if window_start_realtime < 480:
            iso_info[window_count,:] = [session_info['iso_percent_low'], session_info['iso_level_low']] 
        elif (window_start_realtime >= 480) & (window_start_realtime <960):
            iso_info[window_count,:] = [session_info['iso_percent_mid'], session_info['iso_level_mid']] 
        elif window_start_realtime >= 960:
            iso_info[window_count,:] = [session_info['iso_percent_high'], session_info['iso_level_high']] 
                
        #set the start time of the next window in realtime
        diff_end_overlap = window_end_realtime - window_overlap
        window_start_realtime = diff_end_overlap
        
        #calculate how many timepoints are in the overlap after censoring, set start time in censoredtime
        if window_overlap >0:
            num_uncensored_timepoints_overlap = censoring_df_1440length[diff_end_overlap:window_end_realtime].value_counts()[1]
        else:
            num_uncensored_timepoints_overlap = 0
        window_start_censoredtime = window_end_censoredtime - num_uncensored_timepoints_overlap
        window_count = window_count + 1
        
    #now add the information to a dataframe
    df_time = pd.DataFrame(window_time_info, columns=['Start Time Realtime', 'End Time Realtime', 'Start Time Censoredtime',
                                                     'End Time Censoredtime', 'Number of Timepoints'])
    df_iso = pd.DataFrame(iso_info, columns = ['Iso percent', 'Iso val'])
    print(phgy_metrics_mean.shape)
    df_phgy_metrics_mean = pd.DataFrame(phgy_metrics_mean, columns = phgy_df_selectvar.add_suffix('- mean in window').columns)
    df_phgy_metrics_std = pd.DataFrame(phgy_metrics_std, columns = phgy_df_selectvar.add_suffix('- std in window').columns)
    df_cpp_metrics_mean = pd.DataFrame(CPP_metrics_mean, columns = CPP_df.add_suffix('- mean in window').columns)
    df_cpp_metrics_std = pd.DataFrame(CPP_metrics_std, columns = CPP_df.add_suffix('- std in window').columns)
    df_cluster_prediction_counts = pd.DataFrame(cluster_prediction_counts, columns = ['CPP0 frequency (dataset zscore)', 'CPP1 frequency (dataset zscore)', 'CPP2 frequency (dataset zscore)', 'CPP3 frequency (dataset zscore)', 'CPP4 frequency (dataset zscore)',
                                                                                    'CPP0 frequency (sexstrain zscore)', 'CPP1 frequency (sexstrain zscore)', 'CPP2 frequency (sexstrain zscore)', 'CPP3 frequency (sexstrain zscore)', 'CPP4 frequency (sexstrain zscore)',
                                                                                    'CPP0 frequency (sexstrain zscore) sklearn', 'CPP1 frequency (sexstrain zscore) sklearn', 'CPP2 frequency (sexstrain zscore) sklearn', 'CPP3 frequency (sexstrain zscore) sklearn', 'CPP4 frequency (sexstrain zscore) sklearn'])
    df_cluster_prediction_most_freq = pd.DataFrame(cluster_prediction_most_freq, columns = ['most frequent CPP (dataset zscore)', 'most frequent CPP (sexstrain zscore)', 'most frequent CPP (sexstrain zscore) sklearn'])
    df_phgy = pd.concat([df_time, df_iso, df_phgy_metrics_mean, df_phgy_metrics_std, df_cpp_metrics_mean, df_cpp_metrics_std, df_cluster_prediction_counts, df_cluster_prediction_most_freq], axis = 1)

    #save the dataframes as a csv
    df_phgy.to_csv(output_name + "_phgy_windowAvg.csv")
    print('Finished running: ' + output_name)
