#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --account=rrg-mchakrav-ab
module load singularity
singularity run -B /scratch/m/mchakrav/uromil/rabies_runs/mediso-grandjean_yeow-forCAP/bids_niftis:/bids:ro \
-B /scratch/m/mchakrav/uromil/rabies_runs/mediso-grandjean_yeow-forCAP/rabies_out_preprocess-v050_resampled-to-local:/preprocessed_data \
-B /scratch/m/mchakrav/uromil/rabies_runs/mediso-grandjean_yeow-forCAP/rabies_out_cc-v050_resampled-to-local:/confound_corr_data \
/home/m/mchakrav/uromil/rabies-v050.sif -p MultiProc --local_threads 20 confound_correction /preprocessed_data /confound_corr_data \
--image_scaling grand_mean_scaling \
--read_datasink --TR 1.2 \
--conf_list  mot_6 WM_signal CSF_signal vascular_signal \
--frame_censoring FD_censoring=true,FD_threshold=0.05,DVARS_censoring=true,minimum_timepoint=120 \
--highpass 0.01 \
--edge_cutoff 30 \
--smoothing_filter 0.3

#matching to my local confound correction, except add 30s edge cutoff