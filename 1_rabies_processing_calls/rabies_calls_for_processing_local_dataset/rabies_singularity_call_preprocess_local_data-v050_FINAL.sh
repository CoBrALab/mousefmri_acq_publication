#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --account=rrg-mchakrav-ab
singularity run -B /scratch/m/mchakrav/uromil/rabies_runs/local_data/bids_niftis_rest:/bids_niftis:ro \
-B /scratch/m/mchakrav/uromil/rabies_runs/local_data/rabies_out_preprocess-v050:/rabies_out \
/home/m/mchakrav/uromil/rabies-v050.sif -p MultiProc preprocess /bids_niftis /rabies_out --TR 1.0 \
--anat_inho_cor method=SyN,otsu_thresh=2,multiotsu=true \
--anat_robust_inho_cor apply=true,masking=false,brain_extraction=false,template_registration=SyN \
--anatomical_resampling 0.15x0.15x0.15 \
--isotropic_HMC

#don't include commonspace_resampling option because my image resolution is already at the desired resolution of 0.25x0.25x0.5

