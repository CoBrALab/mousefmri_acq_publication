#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --account=rrg-mchakrav-ab
singularity run -B /scratch/m/mchakrav/uromil/rabies_runs/mediso-grandjean_yeow-forCAP/bids_niftis:/bids_niftis:ro \
-B /scratch/m/mchakrav/uromil/rabies_runs/mediso-grandjean_yeow-forCAP/rabies_out_preprocess-v050_resampled-to-local:/rabies_out \
/home/m/mchakrav/uromil/rabies-v050.sif -p MultiProc preprocess /bids_niftis /rabies_out --TR 1.2 \
--anat_inho_cor method=SyN,otsu_thresh=2,multiotsu=true \
--anat_robust_inho_cor apply=true,masking=false,brain_extraction=false,template_registration=SyN \
--anatomical_resampling 0.15x0.15x0.15 \
--commonspace_resampling 0.25x0.5x0.25 \
--isotropic_HMC

#perform commonspace resampling to the same res as my local data
#all other parameters are the same as my local preprocessing param