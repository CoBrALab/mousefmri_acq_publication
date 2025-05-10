#!/usr/bin/env python
# coding: utf-8
# In[1]:


from sklearn.utils import check_random_state
import numpy as np
import nilearn
import nibabel as nb
from nilearn.plotting import plot_stat_map
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
import seaborn as sns
import pandas as pd

#define custom kmeans class and functions for clustering using correlation as a distance - similar format as sklearn kmeans but with the matlab functionality
class KMeansCorrelation:
    def __init__(self, n_clusters, n_rep=1, max_iters=300, tol = 1e-4, threshold = 5, random_state=None):
        self.n_clusters = n_clusters
        self.n_rep = n_rep #the number of replicates is num of times to repeat with new initial cluster positions (like n_init in scikit-learn)
        self.max_iters = max_iters #number of iterations in each replicate
        self.tol = tol #convergence tolerance (Frobenius norm between consecutive centroids)
        self.threshold = threshold
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def correlation_distance(self, X, Y):
        # Ensure X and Y have the same shape
        assert X.shape == Y.shape, "Input matrices must have the same shape"
        # Calculate correlation distance
        return 1 - np.corrcoef(X, Y, rowvar=False)[0, 1]

    def fit(self, data):
        best_labels, best_centroids, best_inertia = None, None, np.inf

        for rep in range(self.n_rep):
            # Randomly initialize centroids with a different seed for each repetition
            rng = np.random.default_rng(seed=self.random_state + rep) if self.random_state is not None else np.random.default_rng()
            centroids = data[rng.choice(data.shape[0], self.n_clusters, replace=False)]

            iter_count = 0 
            centers_squared_diff = 1
            #iteratively update the location of the centroid (from random) until the number of max iterations is reached, or centroid location barely changes (tolerance)
            while (centers_squared_diff > self.tol) and (iter_count < self.max_iters):
                # Assign each data point to the nearest centroid based on correlation distance
                distances = np.array([np.apply_along_axis(lambda x: self.correlation_distance(x, centroid), 1, data) for centroid in centroids])
                labels = np.argmin(distances, axis=0)

                # Update centroids based on the mean of assigned data points
                centroids_new = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])

                #compute the tolerance (how different the new centroids are compared to the previous iteration) using Frobenius norm
                centers_squared_diff = np.sum((centroids_new - centroids) ** 2)
                centroids = centroids_new
                
                #update the iteration number
                print(iter_count)
                iter_count = iter_count + 1
                
                #print the convergence situation
                if centers_squared_diff <= self.tol:
                    print(" Random repetition #" + str(rep) + " converged on iteration " + str(iter_count) + " as tolerance of " + str(centers_squared_diff) + " was reached.")
                elif iter_count == self.max_iters:
                    print(" Random repetition #" + str(rep) + " converged at max iteration " + str(iter_count) + " at tolerance of " + str(centers_squared_diff) + ".")
                else:
                    print("\r","initialization: " + str(rep) + " iteration: " + str(iter_count) + " tolerance: " + str(centers_squared_diff), end="")


            # Calculate within-cluster sums of point-to-centroid distances (inertia)
            inertia = np.sum([np.sum((data[labels == i] - centroids[i])**2) for i in range(self.n_clusters)])

            # Update best solution if the current one has lower inertia
            if inertia < best_inertia:
                best_labels, best_centroids, best_inertia = labels, centroids, inertia
        self.labels_ = best_labels
        self.cluster_centers_ = best_centroids
        return self

    def transform(self, data):
        #to get actual correlation, subtract 1 and divide by -1 from the correlation_distance
        correlations = np.array([np.apply_along_axis(lambda x: (self.correlation_distance(x, centroid)-1.0)/-1.0, 1, data) for centroid in self.cluster_centers_])
        return correlations.T

    def predict(self, data):
        correlations = self.transform(data)
        distances = 1-correlations
        prediction = np.argmin(distances, axis=1)
        print(self.threshold)
        
        #iterate over the clusters, find the timepoints within the lowest correlation percentile of each cluster, assign all these timepoints to a new cluster
        correlations_within_clust = pd.DataFrame(correlations).loc[prediction==0, 0]
        percentile_cutoff_value = np.percentile(correlations_within_clust,self.threshold, axis = 0)
        bool_corr_within_percentile = correlations_within_clust < percentile_cutoff_value

        #also plot the histograms of correlations within each cluster and the percentile
        fig,axs = plt.subplots(nrows=2, ncols=self.n_clusters, figsize=(self.n_clusters*4,7), dpi = 200)
        axs[1,0].hist(1-correlations_within_clust)
        axs[1,0].axvline(1-percentile_cutoff_value, color = 'red')
        axs[1,0].set_xlabel('Distances (1-r)')
        axs[1,0].set_ylabel('Number of timepoints')
        axs[0,0].set_title('Cluster 0 \n' + str(len(correlations_within_clust)) + ' timepoints \n Discard correlations below ' + str(round(percentile_cutoff_value, 4)))
        axs[0,0].hist(correlations_within_clust)
        axs[0,0].axvline(percentile_cutoff_value, color = 'red')
        axs[0,0].set_xlabel('Correlations (r)')
        axs[0,0].set_ylabel('Number of timepoints')

        for clust in range(1,self.n_clusters):
            correlations_within_clust_new = pd.DataFrame(correlations).loc[prediction==clust, clust]
            percentile_cutoff_value = np.percentile(correlations_within_clust_new,self.threshold, axis = 0)
            bool_corr_within_percentile_new = correlations_within_clust_new < percentile_cutoff_value
            
            #also plot the histograms of correlations within each cluster and the percentile
            axs[1,clust].hist(1-correlations_within_clust)
            axs[1,clust].axvline(1-percentile_cutoff_value, color = 'red')
            axs[1,clust].set_xlabel('Distances (1-r)')
            axs[0,clust].set_title('Cluster ' + str(clust) + '\n' + str(len(correlations_within_clust)) + ' timepoints \n Discard correlations below ' + str(round(percentile_cutoff_value, 4)))
            axs[0,clust].hist(correlations_within_clust)
            axs[0,clust].axvline(percentile_cutoff_value, color = 'red')
            axs[0,clust].set_xlabel('Correlations (r)')

            #concatenate results across clusters using index
            correlations_within_clust = pd.concat([correlations_within_clust, correlations_within_clust_new])
            bool_corr_within_percentile = pd.concat([bool_corr_within_percentile, bool_corr_within_percentile_new])

        #sort the final df so indices are in order
        bool_corr_within_percentile = bool_corr_within_percentile.sort_index()
        correlations_within_clust = correlations_within_clust.sort_index()

        #plot final figure
        fig.show()

        #where the dist is in lowest percentile, assign that timepoint to a new cluster
        prediction[bool_corr_within_percentile] = clust + 1
        return correlations_within_clust, prediction

#define function for plotting
def recover_3D(mask_file, vector_map):
    brain_mask = np.asarray(nb.load(mask_file).dataobj)
    volume_indices = brain_mask.astype(bool)
    volume = np.zeros(brain_mask.shape)
    volume[volume_indices] = vector_map
    volume_img = nb.Nifti1Image(volume, nb.load(mask_file).affine, nb.load(mask_file).header)
    return volume_img

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

def extract_array_of_epis(epi_files, commonspace_mask_file):
    #epi_files should be a glob array of files
    epi, mask, epi_masked_flat_init = preprocess_nifti(epi_files[0], commonspace_mask_file, False, 0.3)

    for i in range(1, 5):
        _, mask, epi_masked_flat = preprocess_nifti(epi_files[i], commonspace_mask_file, False, 0.3)

        if i==1:
            all_epi_masked_flat = np.concatenate((epi_masked_flat_init, epi_masked_flat), axis = 1)
        else:
            print(i)
            all_epi_masked_flat = np.concatenate((all_epi_masked_flat, epi_masked_flat), axis = 1)
    return all_epi_masked_flat

def extract_array_of_phgy_with_dict(phgy_files, range_files_to_extract, output_dict_bool):
    phgy_dict = {}
    phgy_shape_dict = {}
    
    #iterate over all csvs and extract an array from each, store outputs in a dict
    for i in range(range_files_to_extract[0], range_files_to_extract[1]):
        num_files_to_extract = range_files_to_extract[1] - range_files_to_extract[0]
        filename = os.path.basename(os.path.abspath(phgy_files[i]))
        ses = filename[2:3]
        sub = filename[4:7]
        sub_ses_ID = 'sub-PHG' + str(sub) + '_ses-' + str(ses)
        
        phgy_df = pd.read_csv(phgy_files[i])
        epi_dict[sub_ses_ID] = phgy_df
        epi_shape_dict[sub_ses_ID] = phgy_df.shape[1]
        #print the progress
        print("\r", 'loading file ' + str(i+1)  + ' out of ' + str(num_files_to_extract), end="")
    
    #concatenate each file stored as an element of the dict
    if output_dict_bool:
        return phgy_dict, phgy_shape_dict
    else:
        all_phgy = np.concatenate([phgy_dict[x] for x in phgy_dict], 1)
        return all_phgy, phgy_shape_dict

def extract_array_of_epis_with_dict(epi_files, commonspace_mask_file, range_files_to_extract, num_timepoints, alt_name, output_dict_bool):
    epi_dict = {}
    epi_shape_dict = {}
    
    #iterate over all niftis and extract an array from each, store outputs in a dict
    for i in range(range_files_to_extract[0], range_files_to_extract[1]):
        num_files_to_extract = range_files_to_extract[1] - range_files_to_extract[0]
        filename = os.path.basename(os.path.abspath(epi_files[i]))
        sub_ses_ID = filename[0:filename.rfind('_task')]
        
        _, mask, epi_masked_flat = preprocess_nifti(os.path.abspath(epi_files[i]), commonspace_mask_file, False, 0.3)
        if alt_name:
            sub_ses_ID = filename[0:filename.rfind('_task')]
            run_ID = filename[filename.rfind('_run'):filename.rfind('_bold')]
            keyname= sub_ses_ID + run_ID
        else:
            keyname = sub_ses_ID
        if num_timepoints is not None:
            epi_dict[keyname] = epi_masked_flat[:,0:num_timepoints]
        else:
            epi_dict[keyname] = epi_masked_flat
            epi_shape_dict[keyname] = epi_masked_flat.shape[1]
        #print the progress
        print("\r", 'loading file ' + str(i+1)  + ' out of ' + str(num_files_to_extract), end="")
    
    #concatenate each file stored as an element of the dict
    if output_dict_bool:
        return epi_dict, epi_shape_dict
    else:
        all_epi_masked_flat = np.concatenate([epi_dict[x] for x in epi_dict], 1)
        return all_epi_masked_flat, epi_shape_dict

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
    
def dice_coefficient(mask1,mask2):
    #this is from rabies/analysis_pkg/analysis_math
    dice = np.sum(mask1*mask2)*2.0 / (np.sum(mask1) + np.sum(mask2))
    return dice

def recover_3D(mask_file, vector_map):
    brain_mask = np.asarray(nb.load(mask_file).dataobj)
    volume_indices = brain_mask.astype(bool)
    volume = np.zeros(brain_mask.shape)
    volume[volume_indices] = vector_map
    volume_img = nb.Nifti1Image(volume, nb.load(mask_file).affine, nb.load(mask_file).header)
    return volume_img
    
def CAP_dice(centroid_img, centroid_mask_pos, network_in_indiv, commonspace_mask_file, smoothing_kernel): 
        
        #smooth the resulting map
        volume_indices=np.asarray(nb.load(commonspace_mask_file).dataobj).astype(bool) 
        network_in_indiv_smoothed = np.array(nilearn.image.smooth_img(recover_3D(commonspace_mask_file,
                                                                         network_in_indiv),
                                                              smoothing_kernel).dataobj)[volume_indices]
        #recover output as 3d array
        network_in_indiv_3d = recover_3D(commonspace_mask_file, network_in_indiv_smoothed).get_fdata()

        #create mask by thresholding
        mean_map_mask_pos, _, _ = percent_masking(network_in_indiv_3d, 0.01, False, 0)

        #calc dice
        dice_coeff = dice_coefficient(mean_map_mask_pos, centroid_img.get_fdata())
        
        return dice_coeff, network_in_indiv_smoothed, mean_map_mask_pos
    
def compute_spatialCorrCovDice_timepoint_to_centroid(timeseries, kmeans_object, centroid_index_list, commonspace_mask_file,
                                                    return_dice_mask_fig_bool, commonspace_template_file):
    ''' I think it's best to apply this directly to the timseries of all epis concatenated,
    then I can also concat subject, iso level etc'''
    num_timepoints = timeseries.shape[1]
    num_centroids = len(centroid_index_list)
    
    #define empty array where correlations will be placed
    spatialCorr = np.zeros((num_timepoints, num_centroids))
    spatialCov = np.zeros((num_timepoints, num_centroids))
    dice = np.zeros((num_timepoints, num_centroids))
    
    #loop over all timepoints and centroids
    centroid_count = 0
    for centroid_index in centroid_index_list:
        centroid_spatial_map = kmeans_object.cluster_centers_[centroid_index, :]
        #find centroid masks
        centroid_img = recover_3D(commonspace_mask_file, kmeans_object.cluster_centers_[centroid_index,:])
        centroid_mask_pos, _, _ = percent_masking(centroid_img.get_fdata(),0.01, False, 0)
        
        for timepoint in range(0, num_timepoints):
            #compute the spatial correlation across voxels
            spatialCorr[timepoint, centroid_count] = stats.spearmanr(centroid_spatial_map, timeseries[:,timepoint])[0]
            spatialCov[timepoint, centroid_count] = np.cov(centroid_spatial_map, timeseries[:,timepoint])[1,0]
            #compute Dice overlap
            dice[timepoint, centroid_count], _, mean_map_mask_pos = CAP_dice(centroid_img, centroid_mask_pos,
                                                                                timeseries[:,timepoint],commonspace_mask_file, 0)
            #if desired (e.g. for QC), return the mask used to compute dice
            if return_dice_mask_fig_bool:
                fig,axs = plt.subplots(nrows=4, ncols=1, figsize=(10,20), dpi = 200)
                plot_stat_map(recover_3D(commonspace_mask_file, timeseries[:,timepoint]),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[0],
                          display_mode='y', colorbar=True, title = 'Subject volume')
                plot_stat_map(nb.Nifti1Image(mean_map_mask_pos,
                                             nb.load(commonspace_mask_file).affine,nb.load(commonspace_mask_file).header),
                              bg_img=commonspace_template_file, axes = axs[1], cut_coords=(0,1,2,3,4,5), display_mode='y', 
                              colorbar=True, title = 'Subject specific mask ')
                plot_stat_map(recover_3D(commonspace_mask_file, kmeans_object.cluster_centers_[centroid_index, :]),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[2],
                          display_mode='y', colorbar=True, title = 'Canonical network')
                plot_stat_map(nb.Nifti1Image(centroid_mask_pos,
                                             nb.load(commonspace_mask_file).affine,nb.load(commonspace_mask_file).header),
                              bg_img=commonspace_template_file, axes = axs[3], cut_coords=(0,1,2,3,4,5), display_mode='y', 
                              colorbar=True, title = 'Canonical mask ')
                fig.suptitle('Dice: ' + str(dice[timepoint, centroid_count]))
                fig.savefig('./Dice_in_timepoint' + str(timepoint) + '-centroid' + str(centroid_index) + '.png')
                plt.close()
     
        centroid_count = centroid_count + 1
    #the output will have the same time dimension as the input timeseries (i.e. already censored)
    return spatialCorr, spatialCov, dice

    
def eval_cluster_prediction_quality_in_window(scanID, epi_flat, commonspace_mask_file, commonspace_template_file,
                                              centroid_index_list, centroid_name_list, dr_df,
                                              predictions_myscan, distances_myscan, spatialCorr_myscan, spatialCov_myscan, dice_myscan,
                                              kmeans_object, path_to_save):
    '''Takes as input one subject timeseries (e.g. from dict where each key is a scan). QC the cluster assignments and
    distance values. Compare them to the DR outputs within the same minute window. Window size will be computed from the provided
    DR csvs.'''
    num_windows = dr_df.shape[0]
    colnames = ['Percent timepoints classified as any network','Mean distance to closest network']
    for centroid_name in ['DMN', 'somatomotor']:
        colnames = colnames + ['Percent timepoints classified ' + str(centroid_name), 
                            'Mean distance to ' + str(centroid_name),
                            'Std distance to ' + str(centroid_name),
                            'Percent timepoints classified ' + str(centroid_name),
                            'Mean distance of classified points to ' + str(centroid_name),
                            'Std distance of classified points to ' + str(centroid_name),
                            'Mean absolute corr to ' + str(centroid_name),
                            'Mean positive corr to ' + str(centroid_name),
                            'Mean negative corr to ' + str(centroid_name),
                            'Percent timepoints with high positive corr to ' + str(centroid_name),
                            'Percent timepoints with high negative corr to ' + str(centroid_name),
                            'Mean absolute cov to ' + str(centroid_name),
                            'Mean positive cov to ' + str(centroid_name),
                            'Mean negative cov to ' + str(centroid_name),
                            'Percent timepoints with high positive cov to ' + str(centroid_name),
                            'Percent timepoints with high negative cov to ' + str(centroid_name),
                            'Mean CAP Dice across timepoints- ' + str(centroid_name),
                            'CAP Dice overlap in window with ' + str(centroid_name)]
    CAP_df = pd.DataFrame(columns=colnames, index=range(1, num_windows))

    for window in range(0,num_windows):
       
        #extract the relevant window from the DR info (could be 2-min or 8-min etc)
        window_start_censoredtime = int(dr_df['Start Time Censoredtime'][window])
        window_start_realtime = int(dr_df['Start Time Realtime'][window])
        window_end_censoredtime = int(dr_df['End Time Censoredtime'][window])
        num_timepoints_window = window_end_censoredtime - window_start_censoredtime
        epi_flat_mywindow = epi_flat[:, window_start_censoredtime : window_end_censoredtime]
        
        #print status
        print("\r", 'analyzing window at ' + str(window_start_realtime)  + ' from scan ' + str(scanID), end="")

        #extract the CAP info on predictions and distances in that window
        predictions_mywindow = predictions_myscan[window_start_censoredtime : window_end_censoredtime]
        distances_mywindow = distances_myscan[window_start_censoredtime : window_end_censoredtime, :]
        spatialCorr_mywindow = spatialCorr_myscan[window_start_censoredtime : window_end_censoredtime]
        spatialCov_mywindow = spatialCov_myscan[window_start_censoredtime : window_end_censoredtime]
        dice_mywindow = dice_myscan[window_start_censoredtime : window_end_censoredtime]
        
        #save the outputs in one figure per window
        fig,axs = plt.subplots(nrows=8*len(centroid_index_list) + 6, ncols=1, figsize=(14,28), dpi = 200)
        
        #iterate over all the networks of interest
        centroid_count = 0
        for centroid in centroid_index_list:
            centroid_name = centroid_name_list[centroid_count]
            ############################## prediction computations #######################################
            #extract indices of timepoints that belong to a given network
            indices = np.where(predictions_mywindow == centroid)[0]
            indices_either_network = np.where(np.logical_or(predictions_mywindow == centroid_index_list[0],
                                                            predictions_mywindow ==centroid_index_list[1]))[0]
            #extract the volumes at these indices
            timepoints_with_network = epi_flat[:, indices]
            #compute the mean spatial map of the timepoints with that network, in that window
            mean_network_from_prediction = np.mean(timepoints_with_network, axis = 1)
            
            ############################# distance computations ########################################
            indices_close_to_network = np.where(distances_mywindow[:,centroid]<200)[0]
            timepoints_close_to_network = epi_flat[:, indices_close_to_network]
            mean_close_network = np.mean(timepoints_close_to_network, axis = 1)
            
            ############################ spatial correlation computations #############################
            indices_corr_to_network = np.where(spatialCorr_mywindow[:,centroid_count]>0.08)[0]
            timepoints_corr_to_network = epi_flat[:, indices_corr_to_network]
            mean_corr_network = np.mean(timepoints_corr_to_network, axis = 1)
            
            indices_corr_to_network_neg = np.where(spatialCorr_mywindow[:,centroid_count]<-0.08)[0]
            timepoints_corr_to_network_neg = epi_flat[:, indices_corr_to_network_neg]
            mean_corr_network_neg = np.mean(timepoints_corr_to_network_neg, axis = 1)
            
            all_pos_corr_indices = np.where(spatialCorr_mywindow[:,centroid_count]>=0)[0]
            all_neg_corr_indices = np.where(spatialCorr_mywindow[:,centroid_count]<=0)[0]
            
            ############################ spatial covariance computations #############################
            indices_cov_to_network = np.where(spatialCov_mywindow[:,centroid_count]>0.035)[0]
            timepoints_cov_to_network = epi_flat[:, indices_cov_to_network]
            mean_cov_network = np.mean(timepoints_cov_to_network, axis = 1)
            indices_cov_to_network_neg = np.where(spatialCov_mywindow[:,centroid_count]<-0.035)[0]
            timepoints_cov_to_network_neg = epi_flat[:, indices_cov_to_network_neg]
            mean_cov_network_neg = np.mean(timepoints_cov_to_network_neg, axis = 1)
            
            all_pos_cov_indices = np.where(spatialCov_mywindow[:,centroid_count]>=0)[0]
            all_neg_cov_indices = np.where(spatialCov_mywindow[:,centroid_count]<=0)[0]
            
            ############################ Dice computations ##############################################
            #find centroid masks
            centroid_img = recover_3D(commonspace_mask_file, kmeans_object.cluster_centers_[centroid,:])
            centroid_mask_pos, _, _ = percent_masking(centroid_img.get_fdata(),0.01, False, 0)
            dice_coeff, mean_network_from_prediction_smoothed, mean_map_mask_pos = CAP_dice(centroid_img, centroid_mask_pos,
                                                                                        mean_network_from_prediction,
                                                                                        commonspace_mask_file, 0.3)
            indices_high_dice = np.where(dice_mywindow[:,centroid_count]>0.1)[0]
            timepoints_high_dice = epi_flat[:, indices_high_dice]
            meanMap_high_dice_timepoints = np.mean(timepoints_high_dice, axis = 1)
            
            ############################ save metrics in each window ##################################
                   #save the number of timepoints classified as dmn or somatomotor or either in the window
            CAP_df.at[window, 'Percent timepoints classified ' + str(centroid_name)] = indices.shape[0]/num_timepoints_window
            CAP_df.at[window, 'Percent timepoints classified as any network'] = indices_either_network.shape[0]/num_timepoints_window

            #save the mean and std of the distances to the centroid, in that window, also plot hist
            CAP_df.at[window, 'Mean distance to ' + str(centroid_name)] = np.mean(distances_mywindow[:,centroid])
            CAP_df.at[window, 'Std distance to ' + str(centroid_name)] = np.std(distances_mywindow[:,centroid]) 
            CAP_df.at[window, 'Mean distance to closest network'] = np.mean(np.minimum(distances_mywindow[:,centroid_index_list[0]], 
                                                            distances_mywindow[:,centroid_index_list[1]])) #closest to either

            #save the mean and std of the distances to each network (of timepoints in that CAP)
            CAP_df.at[window, 'Mean distance of classified points to ' + str(centroid_name)] = np.mean(distances_mywindow[indices, centroid])
            CAP_df.at[window, 'Std distance of classified points to ' + str(centroid_name)] = np.std(distances_mywindow[indices, centroid])

            #find number of timepoints with spatial correlation above threshold
            CAP_df.at[window, 'Mean absolute corr to ' + str(centroid_name)] = np.mean(np.abs(spatialCorr_mywindow[:,centroid_count]))
            CAP_df.at[window, 'Mean positive corr to ' + str(centroid_name)] = np.mean(spatialCorr_mywindow[all_pos_corr_indices,
                                                                                                              centroid_count])
            CAP_df.at[window, 'Mean negative corr to ' + str(centroid_name)] = np.mean(spatialCorr_mywindow[all_neg_corr_indices,
                                                                                                              centroid_count])
            CAP_df.at[window, 'Percent timepoints with high positive corr to ' + str(centroid_name)] = indices_corr_to_network.shape[0]/num_timepoints_window
            
            CAP_df.at[window, 'Percent timepoints with high negative corr to ' + str(centroid_name)] = indices_corr_to_network_neg.shape[0]/num_timepoints_window
            
            #find number of timepoints with spatial covariance above threshold
            CAP_df.at[window, 'Mean absolute cov to ' + str(centroid_name)] = np.mean(np.abs(spatialCov_mywindow[:,centroid_count]))
            CAP_df.at[window, 'Mean positive cov to ' + str(centroid_name)] = np.mean(spatialCov_mywindow[all_pos_cov_indices,
                                                                                                              centroid_count])
            CAP_df.at[window, 'Mean negative cov to ' + str(centroid_name)] = np.mean(spatialCorr_mywindow[all_neg_cov_indices,
                                                                                                              centroid_count])
            CAP_df.at[window, 'Percent timepoints with high positive cov to ' + str(centroid_name)] = indices_cov_to_network.shape[0]/num_timepoints_window
            
            CAP_df.at[window, 'Percent timepoints with high negative cov to ' + str(centroid_name)] = indices_cov_to_network_neg.shape[0]/num_timepoints_window
            
            #find mean spatial map of all timepoints classified as either network, take Dice with centroid
            CAP_df.at[window, 'CAP Dice overlap in window with ' + str(centroid_name)] = dice_coeff
            
            #mean of dice values at each timepoint
            CAP_df.at[window, 'Mean CAP Dice across timepoints- ' + str(centroid_name)] = np.mean(dice_mywindow[:,centroid_count])
            ########################## generate QC figure #############################################
            
            #plot the mean spatial map of timepoints predicted to have that network, smoothed version
            plot_stat_map(recover_3D(commonspace_mask_file, mean_network_from_prediction_smoothed),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[0+ centroid_count*8],
                          display_mode='y', colorbar=True, 
                          title = 'Mean predicted as ' + str(centroid_name_list[centroid_count]) + ', smoothed')
            #Mask of above after thresholding to top 1%
            plot_stat_map(nb.Nifti1Image(mean_map_mask_pos, nb.load(commonspace_mask_file).affine,
                                          nb.load(commonspace_mask_file).header),
                           bg_img=commonspace_template_file, axes = axs[1+ centroid_count*8], cut_coords=(0,1,2,3,4,5), 
                           display_mode='y', colorbar=True, 
                          title = 'Mask predicted as '+ str(centroid_name_list[centroid_count]))
            #plot the mean spatial map of timepoints close to that network
            plot_stat_map(recover_3D(commonspace_mask_file, mean_close_network),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[2+ centroid_count*8],
                          display_mode='y', colorbar=True,
                          title = 'Distance<150 to ' + str(centroid_name_list[centroid_count]) + '-' + str(len(indices_close_to_network)))
             #plot the mean spatial map of timepoints highly correlated with that network
            plot_stat_map(recover_3D(commonspace_mask_file, mean_corr_network),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[3+ centroid_count*8],
                          display_mode='y', colorbar=True,
                          title = 'Corr>0.08 to ' + str(centroid_name_list[centroid_count]) + '-' + str(len(indices_corr_to_network)))
            plot_stat_map(recover_3D(commonspace_mask_file, mean_corr_network_neg),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[4+ centroid_count*8],
                          display_mode='y', colorbar=True,
                          title = 'Corr<-0.08 to ' + str(centroid_name_list[centroid_count]) + '-' + str(len(indices_corr_to_network_neg)))
             #plot the mean spatial map of timepoints highly covaried with that network
            plot_stat_map(recover_3D(commonspace_mask_file, mean_cov_network),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[5+ centroid_count*8],
                          display_mode='y', colorbar=True,
                          title = 'Cov>0.035 to ' + str(centroid_name_list[centroid_count]) + '-' + str(len(indices_cov_to_network)))
            plot_stat_map(recover_3D(commonspace_mask_file, mean_cov_network_neg),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[6+ centroid_count*8],
                          display_mode='y', colorbar=True,
                          title = 'Cov<-0.035 to ' + str(centroid_name_list[centroid_count]) + '-' + str(len(indices_cov_to_network_neg)))
            #plot mean spatial map of timepoints with high dice
            plot_stat_map(recover_3D(commonspace_mask_file, meanMap_high_dice_timepoints),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[7+ centroid_count*8],
                          display_mode='y', colorbar=True,
                          title = 'Dice>0.1 to ' + str(centroid_name_list[centroid_count]) + '-' + str(len(indices_cov_to_network_neg)))

                        
            #plot the distance and prediction and correlation timecourses
            axs[8*len(centroid_index_list)].plot(distances_mywindow[:,centroid], label = centroid_name_list[centroid_count])
            axs[8*len(centroid_index_list) + 1].plot(predictions_mywindow == centroid, label = centroid_name_list[centroid_count])
            axs[8*len(centroid_index_list) + 2].plot(spatialCorr_mywindow[:,centroid_count], label = centroid_name_list[centroid_count])
            axs[8*len(centroid_index_list) + 3].plot(spatialCov_mywindow[:,centroid_count], label = centroid_name_list[centroid_count])
            axs[8*len(centroid_index_list) + 4].plot(dice_mywindow[:,centroid_count], label = centroid_name_list[centroid_count])
            axs[8*len(centroid_index_list)].legend()
            axs[8*len(centroid_index_list)].set_ylabel('Distances')
            axs[8*len(centroid_index_list) + 1].set_ylabel(r'Classified as ' + "\n" + 'a given network?')
            axs[8*len(centroid_index_list) + 2].set_ylabel('Correlations')
            axs[8*len(centroid_index_list) + 3].set_ylabel('Covariance')
            axs[8*len(centroid_index_list) + 4].set_ylabel('Dice')
            
            #increment centroid
            centroid_count = centroid_count + 1
            
        if len(centroid_index_list) == 2:
            axs[8*len(centroid_index_list) + 5].plot(np.logical_or(predictions_mywindow == centroid_index_list[0], predictions_mywindow ==centroid_index_list[1]))
            axs[8*len(centroid_index_list) + 5].set_ylabel(r'Classified as' + '\n' +  'any network?')

        plt.suptitle("The DR somatomotor dice is: " + str(dr_df['Dice Prior1'][window]), fontsize = 25)
        plt.savefig(path_to_save + '_' + str(window_start_realtime) + 's.png')
        plt.close()
    ###################################### save final df ##########################3
    sub_id_df = pd.DataFrame([scanID] * num_windows, columns = ['subject'])
    final_df = pd.concat([sub_id_df, dr_df, CAP_df], axis = 1)
    final_df.to_csv(path_to_save + '_metrics.csv')
    print(final_df)
    return final_df, CAP_df

def eval_custom_cluster_prediction_quality_in_window(scanID, epi_censored, commonspace_mask_file, commonspace_template_file,
                                              centroid_index_list, centroid_name_list, dr_df, cpp_metrics,
                                              prediction, correlations,
                                              path_to_save, scan_start_index, scan_end_index):
    '''Takes as input one subject timeseries (e.g. from dict where each key is a scan). QC the cluster assignments and
    distance values. Compare them to the DR outputs within the same minute window. Window size will be computed from the provided
    DR csvs.'''
    num_windows = dr_df.shape[0]
    predictions_myscan = prediction[scan_start_index: scan_end_index]
    correlations_myscan = correlations[scan_start_index: scan_end_index, :]
    avg_correlations_myscan = (np.abs(correlations_myscan[:,1]) + np.abs(correlations_myscan[:,0]))/2
    indices_higher_dmn = np.where(correlations_myscan[:,1] <= correlations_myscan[:,0])
    avg_correlations_myscan[indices_higher_dmn[0]] = (-1)*avg_correlations_myscan[indices_higher_dmn[0]]

    epi_censored_myscan = epi_censored.to_numpy()
    realtimes_myscan = cpp_metrics['Time after scan start'][scan_start_index: scan_end_index].reset_index(drop = True)
    distances_myscan = 1-correlations_myscan
    colnames = ['Percent timepoints classified as any network','Mean distance to closest network', 'Mean avg corr to networks']
    for centroid_name in ['DMN', 'somatomotor']:
        colnames = colnames + ['Percent timepoints classified ' + str(centroid_name), 
                            'Mean distance to ' + str(centroid_name),
                            'Std distance to ' + str(centroid_name),
                            'Mean distance of classified points to ' + str(centroid_name),
                            'Std distance of classified points to ' + str(centroid_name),
                            'Mean absolute corr to ' + str(centroid_name),
                            'Mean positive corr to ' + str(centroid_name),
                            'Mean negative corr to ' + str(centroid_name),
                            'Percent timepoints with high positive corr to ' + str(centroid_name),
                            'Percent timepoints with high negative corr to ' + str(centroid_name)]
    CAP_df = pd.DataFrame(columns=colnames, index=range(1, num_windows))

    for window in range(0,num_windows):
        #extract the relevant window from the DR info (could be 2-min or 8-min etc)
        window_start_realtime = int(dr_df['Start Time Realtime'][window])
        window_end_realtime = int(dr_df['End Time Realtime'][window])

        arr_of_censored_timepoint_indices_in_window = np.where((realtimes_myscan >=  window_start_realtime) & (realtimes_myscan <  window_end_realtime))[0]
        window_start_censoredtime = arr_of_censored_timepoint_indices_in_window[0]
        window_end_censoredtime = arr_of_censored_timepoint_indices_in_window[-1]
        num_timepoints_window = len(arr_of_censored_timepoint_indices_in_window)
                
        #print status
        #print("\r", 'analyzing window at ' + str(window_start_realtime)  + ' from scan ' + str(scanID), end="")
        print('realtime: ' + str(window_start_realtime)  + '-' + str(window_end_realtime) + ', censored time: ' + str(window_start_censoredtime)  + '-' + str(window_end_censoredtime) + ', actual realtime: ' + str(realtimes_myscan[window_start_censoredtime])  + '-' + str(realtimes_myscan[window_end_censoredtime]) +' with timepoint num ' + str(num_timepoints_window) + ' from scan ' + str(scanID))
        
        #extract the CAP info on predictions and distances in that window
        predictions_mywindow = predictions_myscan[window_start_censoredtime : window_end_censoredtime]
        distances_mywindow = distances_myscan[window_start_censoredtime : window_end_censoredtime, :]
        spatialCorr_mywindow = correlations_myscan[window_start_censoredtime : window_end_censoredtime, :]
        epi_censored_mywindow = epi_censored_myscan[:, window_start_censoredtime:window_end_censoredtime]
        avg_spatialCorr_mywindow = avg_correlations_myscan[window_start_censoredtime:window_end_censoredtime]

        #save the outputs in one figure per window
        fig,axs = plt.subplots(nrows=3*len(centroid_index_list) + 4, ncols=1, figsize=(14,20), dpi = 200)
        
        #iterate over all the networks of interest
        centroid_count = 0
        for centroid in centroid_index_list:
            centroid_name = centroid_name_list[centroid_count]
            ############################## prediction computations #######################################
            #extract indices of timepoints that belong to a given network
            indices = np.where(predictions_mywindow == centroid)[0]
            indices_either_network = np.where(np.logical_or(predictions_mywindow == centroid_index_list[0],
                                                            predictions_mywindow ==centroid_index_list[1]))[0]
            #extract the volumes at these indices
            timepoints_with_network = epi_censored_mywindow[:, indices]
            #compute the mean spatial map of the timepoints with that network, in that window
            mean_network_from_prediction = np.mean(timepoints_with_network, axis = 1)
            
            ############################ spatial correlation computations #############################
            indices_corr_to_network = np.where(spatialCorr_mywindow[:,centroid_count]>0.2)[0]
            timepoints_corr_to_network = epi_censored_mywindow[:, indices_corr_to_network]
            mean_corr_network = np.mean(timepoints_corr_to_network, axis = 1)
            
            indices_corr_to_network_neg = np.where(spatialCorr_mywindow[:,centroid_count]<-0.2)[0]
            timepoints_corr_to_network_neg = epi_censored_mywindow[:, indices_corr_to_network_neg]
            mean_corr_network_neg = np.mean(timepoints_corr_to_network_neg, axis = 1)
            
            all_pos_corr_indices = np.where(spatialCorr_mywindow[:,centroid_count]>=0)[0]
            all_neg_corr_indices = np.where(spatialCorr_mywindow[:,centroid_count]<=0)[0]
            
            
            ############################ save metrics in each window ##################################
                   #save the number of timepoints classified as dmn or somatomotor or either in the window
            CAP_df.at[window, 'Percent timepoints classified ' + str(centroid_name)] = indices.shape[0]/num_timepoints_window
            CAP_df.at[window, 'Percent timepoints classified as any network'] = indices_either_network.shape[0]/num_timepoints_window
            CAP_df.at[window, 'Mean avg corr to networks'] = np.mean(np.abs(avg_spatialCorr_mywindow))
            #save the mean and std of the distances to the centroid, in that window, also plot hist
            CAP_df.at[window, 'Mean distance to ' + str(centroid_name)] = np.mean(distances_mywindow[:,centroid])
            CAP_df.at[window, 'Std distance to ' + str(centroid_name)] = np.std(distances_mywindow[:,centroid]) 
            CAP_df.at[window, 'Mean distance to closest network'] = np.mean(np.minimum(distances_mywindow[:,centroid_index_list[0]], 
                                                            distances_mywindow[:,centroid_index_list[1]])) #closest to either

            #save the mean and std of the distances to each network (of timepoints in that CAP)
            CAP_df.at[window, 'Mean distance of classified points to ' + str(centroid_name)] = np.mean(distances_mywindow[indices, centroid])
            CAP_df.at[window, 'Std distance of classified points to ' + str(centroid_name)] = np.std(distances_mywindow[indices, centroid])
            #find number of timepoints with spatial correlation above threshold
            CAP_df.at[window, 'Mean absolute corr to ' + str(centroid_name)] = np.mean(np.abs(spatialCorr_mywindow[:,centroid_count]))
            CAP_df.at[window, 'Mean positive corr to ' + str(centroid_name)] = np.mean(spatialCorr_mywindow[all_pos_corr_indices,
                                                                                                              centroid_count])
            CAP_df.at[window, 'Mean negative corr to ' + str(centroid_name)] = np.mean(spatialCorr_mywindow[all_neg_corr_indices,
                                                                                                              centroid_count])
            CAP_df.at[window, 'Percent timepoints with high positive corr to ' + str(centroid_name)] = indices_corr_to_network.shape[0]/num_timepoints_window
            
            CAP_df.at[window, 'Percent timepoints with high negative corr to ' + str(centroid_name)] = indices_corr_to_network_neg.shape[0]/num_timepoints_window
            
            ########################## generate QC figure #############################################
            
            #plot the mean spatial map of timepoints predicted to have that network, smoothed version
            plot_stat_map(recover_3D(commonspace_mask_file, mean_network_from_prediction),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[0+ centroid_count*3],
                          display_mode='y', colorbar=True, 
                          title = 'Mean predicted as ' + str(centroid_name_list[centroid_count]) + ', smoothed')
             #plot the mean spatial map of timepoints highly correlated with that network
            plot_stat_map(recover_3D(commonspace_mask_file, mean_corr_network),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[1+ centroid_count*3],
                                                    display_mode='y', colorbar=True,
                          title = 'Corr>0.2 to ' + str(centroid_name_list[centroid_count]) + '-' + str(len(indices_corr_to_network)))
            plot_stat_map(recover_3D(commonspace_mask_file, mean_corr_network_neg),
                          bg_img=commonspace_template_file, cut_coords=(0,1,2,3,4,5), axes = axs[2+ centroid_count*3],
                          display_mode='y', colorbar=True,
                          title = 'Corr<-0.2 to ' + str(centroid_name_list[centroid_count]) + '-' + str(len(indices_corr_to_network_neg)))
            #plot the distance and prediction and correlation timecourses
            axs[3*len(centroid_index_list)].plot(distances_mywindow[:,centroid], label = centroid_name_list[centroid_count])
            axs[3*len(centroid_index_list) + 1].plot(predictions_mywindow == centroid, label = centroid_name_list[centroid_count])
            axs[3*len(centroid_index_list) + 2].plot(spatialCorr_mywindow[:,centroid_count], label = centroid_name_list[centroid_count])
            axs[3*len(centroid_index_list)].legend()
            axs[3*len(centroid_index_list)].set_ylabel('Distances')
            axs[3*len(centroid_index_list) + 1].set_ylabel(r'Classified as ' + "\n" + 'a given network?')
            axs[3*len(centroid_index_list) + 2].set_ylabel('Correlations')
            
            #increment centroid
            centroid_count = centroid_count + 1
            
        if len(centroid_index_list) == 2:
            axs[3*len(centroid_index_list) + 3].plot(np.logical_or(predictions_mywindow == centroid_index_list[0], predictions_mywindow ==centroid_index_list[1]))
            axs[3*len(centroid_index_list) + 3].set_ylabel(r'Classified as' + '\n' +  'any network?')

        plt.suptitle("The DR somatomotor corr is: " + str(dr_df['Correlation Somatomotor'][window]) + '\n actual realtime: ' + str(realtimes_myscan[window_start_censoredtime])  + '-' + str(realtimes_myscan[window_end_censoredtime]) + ', num timepoints: ' + str(num_timepoints_window), fontsize = 25)
        plt.savefig(path_to_save + '_' + str(window_start_realtime) + 's.png')
        plt.close()
        ###################################### save final df ##########################3
    sub_id_df = pd.DataFrame([scanID] * num_windows, columns = ['subject'])
    final_df = pd.concat([sub_id_df, dr_df, CAP_df], axis = 1)
    final_df.to_csv(path_to_save + '_metrics.csv')
    return final_df, CAP_df

def create_CAP_timepoint_sanitycheck_figs_fulldata(scan, epi_censored, all_cap_metrics, all_cpp_metrics,
                                           commonspace_mask_file_local,commonspace_template_file_local,
                                             scan_start_index, scan_end_index, dict_network):
        
        predictions_myscan = all_cap_metrics['Prediction'][scan_start_index: scan_end_index]
        epi_local_flat_myscan = epi_censored.to_numpy()
        corr_myscan = all_cap_metrics['Corr-Somatomotor'][scan_start_index: scan_end_index]
        corr_myscan_dmn = all_cap_metrics['Corr-DMN'][scan_start_index: scan_end_index]
        realtime = all_cpp_metrics['Time after scan start'][scan_start_index: scan_end_index]
        timepoint = scan_start_index
        while (timepoint<scan_end_index):
            print("\r",
                  'timepoint: ' + str(timepoint) + ', timepoint realtime: ' + str(realtime[timepoint]) + ', timepoint-scan_start_time: ' + str(timepoint-scan_start_index) + ', (timepoint-scan_start_index)%10: ' + str((timepoint-scan_start_index)%10)
                  , end="")
            #don't need ALL the outputs per subject
            if (realtime[timepoint] <100) or (480<=realtime[timepoint]<550) or (960<=realtime[timepoint]<1030):
                #every 10 frames, initialize a new figure
                if (timepoint-scan_start_index)%10 == 0:
                    fig,axs = plt.subplots(nrows=10, ncols=3, figsize=(18,18), dpi = 200)
                    axs[0, 1].set_title('Corr to somat', fontsize = 12)
                    axs[0, 2].set_title('Corr to DMN', fontsize = 12)

                #plot each (non-censored timepoint)
                timepoint_title = str('Timepoint ' + str(realtime[timepoint]) + ' cluster ' + str(dict_network[str(predictions_myscan[timepoint])]) ) 
                plot_stat_map(recover_3D(commonspace_mask_file_local, epi_local_flat_myscan[:,timepoint]),
                            bg_img=nb.load(commonspace_template_file_local), cut_coords=(0,1,2,3,4,5), axes = axs[(timepoint-scan_start_index)%10,0],
                            display_mode='y', colorbar=False, vmax = 3.5, annotate = False)
                
                #plot cluster membership and correlation strength
                if (predictions_myscan[timepoint] == 0) or (predictions_myscan[timepoint] == 1):
                    value_somat = [corr_myscan[timepoint]]
                    value_dmn = [corr_myscan_dmn[timepoint]]
                else:
                    value_somat = 0.000
                    value_dmn = 0.000
                #set the border color to black for the cluster that a given timepoint was assigned to
                dict_box_boundary_ax1 = {'0': 'white', '1': 'black', '2': 'white', '3': 'white'}
                dict_box_boundary_ax2 = {'0': 'black', '1': 'white', '2': 'white', '3': 'white'}
                sns.heatmap(np.array(value_somat).reshape((1,1)), vmax = 0.563, vmin = -0.563, cmap = 'seismic', 
                            linecolor = dict_box_boundary_ax1[str(predictions_myscan[timepoint])], linewidths = 2, 
                            clip_on = False, annot= True, annot_kws = {'fontsize': 20}, xticklabels = False, yticklabels = False,
                            ax = axs[(timepoint-scan_start_index)%10, 1])
                sns.heatmap(np.array(value_dmn).reshape((1,1)), vmax = 0.563, vmin = -0.563, cmap = plt.cm.get_cmap('seismic').reversed(), 
                            linecolor = dict_box_boundary_ax2[str(predictions_myscan[timepoint])], linewidths = 2, 
                            clip_on = False,annot= True, annot_kws = {'fontsize': 20}, xticklabels = False, yticklabels = False,
                            ax = axs[(timepoint-scan_start_index)%10, 2])
                axs[timepoint%10, 0].set_title(timepoint_title, fontsize = 12)
                
                #on the last of the 10 frames, save the image
                if (timepoint-scan_start_index)%10 == 9:
                    fig.savefig('./sanitycheck_CAP_timepoint_level/CAP_timepoint_' + str(scan) + '-start' + str(timepoint-scan_start_index-9) + '.png')
                    plt.close()
            timepoint = timepoint + 1
def create_CAP_timepoint_sanitycheck_figs_onesub(scan, epi_censored, all_cap_metrics, all_cpp_metrics,
                                           commonspace_mask_file_local,commonspace_template_file_local,
                                             scan_start_index, scan_end_index, dict_network):
        
        predictions_myscan = all_cap_metrics['Prediction'][scan_start_index: scan_end_index].reset_index(drop=True)
        epi_local_flat_myscan = epi_censored.to_numpy()
        corr_myscan = all_cap_metrics['Corr-Somatomotor'][scan_start_index: scan_end_index].reset_index(drop=True)
        corr_myscan_dmn = all_cap_metrics['Corr-DMN'][scan_start_index: scan_end_index].reset_index(drop=True)
        realtime = all_cpp_metrics['Time after scan start'][scan_start_index: scan_end_index].reset_index(drop = True)
        timepoint = 0

        #take the average correlation between somat and dmn, then set the sign to negative if that timepoint is more dmn-like
        corr_myscan_avg_somatdmn = (np.abs(corr_myscan) + np.abs(corr_myscan_dmn))/2
        indices_higher_dmn = np.where(corr_myscan <= corr_myscan_dmn)
        corr_myscan_avg_somatdmn[indices_higher_dmn[0]] = (-1)*corr_myscan_avg_somatdmn[indices_higher_dmn[0]]
        while (timepoint<epi_local_flat_myscan.shape[1]):
            print("\r",
                  'timepoint: ' + str(timepoint) + ', timepoint realtime: ' + str(realtime[timepoint]) + ', timepoint%10: ' + str((timepoint)%10)
                  , end="")
            #don't need ALL the outputs per subject
            if (realtime[timepoint] <100) or (480<=realtime[timepoint]<550) or (960<=realtime[timepoint]<1030):
                #every 10 frames, initialize a new figure
                if (timepoint)%10 == 0:
                    fig,axs = plt.subplots(nrows=10, ncols=4, figsize=(22,18), dpi = 200)
                    axs[0, 1].set_title('Corr to somat', fontsize = 12)
                    axs[0, 2].set_title('Corr to DMN', fontsize = 12)
                    axs[0, 3].set_title('Average corr to network', fontsize = 12)

                #plot each (non-censored timepoint)
                timepoint_title = str('Timepoint ' + str(realtime[timepoint]) + ' cluster ' + str(dict_network[str(predictions_myscan[timepoint])]) ) 
                plot_stat_map(recover_3D(commonspace_mask_file_local, epi_local_flat_myscan[:,timepoint]),
                            bg_img=nb.load(commonspace_template_file_local), cut_coords=(0,1,2,3,4,5), axes = axs[(timepoint)%10,0],
                            display_mode='y', colorbar=False, vmax = 3.5, annotate = False)
                
                #plot cluster membership and correlation strength
                if (predictions_myscan[timepoint] == 0) or (predictions_myscan[timepoint] == 1):
                    value_somat = [corr_myscan[timepoint]]
                    value_dmn = [corr_myscan_dmn[timepoint]]
                else:
                    value_somat = 0.000
                    value_dmn = 0.000
                value_avg = [corr_myscan_avg_somatdmn[timepoint]]
                #set the border color to black for the cluster that a given timepoint was assigned to
                dict_box_boundary_ax1 = {'0': 'white', '1': 'black', '2': 'white', '3': 'white'}
                dict_box_boundary_ax2 = {'0': 'black', '1': 'white', '2': 'white', '3': 'white'}
                sns.heatmap(np.array(value_somat).reshape((1,1)), vmax = 0.563, vmin = -0.563, cmap = 'seismic', 
                            linecolor = dict_box_boundary_ax1[str(predictions_myscan[timepoint])], linewidths = 2, 
                            clip_on = False, annot= True, annot_kws = {'fontsize': 20}, xticklabels = False, yticklabels = False,
                            ax = axs[(timepoint)%10, 1])
                sns.heatmap(np.array(value_dmn).reshape((1,1)), vmax = 0.563, vmin = -0.563, cmap = plt.cm.get_cmap('seismic').reversed(), 
                            linecolor = dict_box_boundary_ax2[str(predictions_myscan[timepoint])], linewidths = 2, 
                            clip_on = False,annot= True, annot_kws = {'fontsize': 20}, xticklabels = False, yticklabels = False,
                            ax = axs[(timepoint)%10, 2])
                sns.heatmap(np.array(value_avg).reshape((1,1)), vmax = 0.563, vmin = -0.563, cmap = plt.cm.get_cmap('seismic'), 
                            clip_on = False,annot= True, annot_kws = {'fontsize': 20}, xticklabels = False, yticklabels = False,
                            ax = axs[(timepoint)%10, 3])
                axs[timepoint%10, 0].set_title(timepoint_title, fontsize = 12)
                
                #on the last of the 10 frames, save the image
                if (timepoint)%10 == 9:
                    fig.savefig('./sanitycheck_CAP_timepoint_level/CAP_timepoint_' + str(scan) + '-start' + str(realtime[timepoint-9]) + '.png')
                    plt.close()
            timepoint = timepoint + 1