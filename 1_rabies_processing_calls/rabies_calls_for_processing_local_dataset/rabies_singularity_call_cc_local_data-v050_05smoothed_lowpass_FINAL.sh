#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --account=rrg-mchakrav-ab
module load singularity
singularity run -B /scratch/m/mchakrav/uromil/rabies_runs/local_data/bids_niftis_rest:/bids:ro \
-B /scratch/m/mchakrav/uromil/rabies_runs/local_data/rabies_out_preprocess-v050:/preprocessed_data \
-B /scratch/m/mchakrav/uromil/rabies_runs/local_data/rabies_out_cc-v050_05smoothed_lowpass:/confound_corr_data \
/home/m/mchakrav/uromil/rabies-v050.sif -p MultiProc --local_threads 20 confound_correction /preprocessed_data /confound_corr_data \
--image_scaling grand_mean_scaling \
--read_datasink --TR 1.0 \
--conf_list  mot_6 WM_signal CSF_signal vascular_signal \
--frame_censoring FD_censoring=true,FD_threshold=0.05,DVARS_censoring=true,minimum_timepoint=120 \
--highpass 0.01 \
--lowpass 0.3 \
--edge_cutoff 30 \
--generate_CR_null \
--smoothing_filter 0.5

#I decided to smooth by final outputs, because that's what I have to do anyways for comparison with the RABIES paper
#the only difference will be that I am not resampling my local data to the common resolution