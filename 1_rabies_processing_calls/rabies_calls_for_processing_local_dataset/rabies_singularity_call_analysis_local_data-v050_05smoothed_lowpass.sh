#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --account=rrg-mchakrav-ab
module load singularity
singularity run -B /scratch/m/mchakrav/uromil/rabies_runs/local_data/bids_niftis_rest:/bids:ro \
-B /scratch/m/mchakrav/uromil/rabies_runs/local_data/rabies_out_preprocess-v050:/preprocessed_data \
-B /scratch/m/mchakrav/uromil/rabies_runs/local_data/rabies_out_cc-v050_05smoothed_lowpass:/confound_corr_data \
-B /scratch/m/mchakrav/uromil/rabies_runs/local_data/rabies_out_analysis-v050_05smoothed_lowpass:/analysis_outputs \
/home/m/mchakrav/uromil/rabies-v050.sif -p MultiProc --local_threads 10 --figure_format svg analysis /confound_corr_data /analysis_outputs \
--data_diagnosis
