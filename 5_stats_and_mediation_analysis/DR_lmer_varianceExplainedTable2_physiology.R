#load libraries
library(dplyr)
library(stringi)
library(fastDummies)
library(lme4)
library(effects)
library(ggplot2)
library(lmerTest)
library(psych)
library(sjPlot)
library(jtools)
library(performance)
library(brms)
library(sjstats)
library(patchwork)
library(bayestestR)
library(relaimpo)
library(gt)
library(webshot2)

# the purpose of this script is to a) obtain the effect size estimates for each effect in the mediation analysis - these will be used for the stick plots on the side of the scatterplots
# b) generate the variance explained tables.
# these two purposes are grouped together because they both rely on running lmer models

#load DR outputs
DR_df=read.csv('../../DR_analysis-to_extract_network_detectability/final_outputs/master_DR_and_phgy_variableWindow_sparse.csv')

#fisherz
DR_df$Average.correlation.to.network.fisherz <- psych::fisherz(DR_df$Average.correlation.to.network)
write.csv(DR_df, "../mediation_path_effectSize_csvs/master_DR_and_phgy_variableWindow_sparse_fisherzND.csv")

#set sub, sex and strain as factors. Iso, dex and session_order also need to be set as factors in order to be recoded later.
DR_df$subject_ID<- as.factor(DR_df$subject_ID)
DR_df$strain <- factor(DR_df$strain, levels = c('C57Bl/6', 'C3HeB/FeJ')) #specify the order so that C57 is baseline (reference)
DR_df$sex <-factor(DR_df$sex, levels = c('m', 'f'))
DR_df$isoflurane_percent <- as.factor(DR_df$Iso.percent)
DR_df$dex_conc <- as.factor(DR_df$dex_conc)
DR_df$actual_ses_order <- as.factor(DR_df$actual_ses_order)

summary(DR_df)

#remove the datapoints corresponding to iso=1.5% because it is relatively few (<3000) compared to 14,000+ for other iso levels, and might be confusing
df<-droplevels(subset(DR_df, isoflurane_percent != 1.5)) #this removed 5% of the datapoints

#recode iso and dex using backward difference coding scheme (because they have uniform distributions rn) - this coding scheme will compare each level to the previous level instead of to a single reference
backward_diff_3_contrast_matrix <- MASS::contr.sdif(3) #define the matrix of how each iso value gets recoded
contrasts(df$isoflurane_percent) = backward_diff_3_contrast_matrix #somehow assign these new values to the original iso levels
contrasts(df$dex_conc) = backward_diff_3_contrast_matrix
contrasts(df$actual_ses_order) = backward_diff_3_contrast_matrix

#abs of the correlation values (b/c a strong negative correlation to somatomotor means high DMN detectability)
df$network_detectability <- log(psych::fisherz(df$Average.correlation.to.network)) #fisherz accounts for fact that going from 0.01 to 0.02 correlation is more likely than from 0.91 to 0.92 (just by sampling some more you can increase corr a little bit when its already low)

#standardize the necessary variables
continuous_dependent_variables = c('Mean.FD..mean.in.window', 'Max.FD..mean.in.window', 'Mean.FD..std.in.window', 'Max.FD..std.in.window')
other_bounded_variables = c('weight', 'Start.Time.Realtime', 'age_days', 'Time.after.isoflurane.change', 'Number.of.Timepoints')
phgy_variables = c('RR..mean.in.window', 'RRV..mean.in.window', 'RV..mean.in.window', 'HR..mean.in.window', 'HRV..mean.in.window', 'PVI..mean.in.window', 'SpO2..mean.in.window', 'RR..std.in.window', 'RRV..std.in.window', 'RV..std.in.window', 'HR..std.in.window', 'HRV..std.in.window', 'PVI..std.in.window', 'SpO2..std.in.window')
df_scaled <-df %>% 
  dplyr::mutate(across(starts_with(continuous_dependent_variables), standardize)) %>%
  dplyr:::mutate(across(starts_with(other_bounded_variables), standardize)) %>%
  dplyr:::mutate(across(starts_with(phgy_variables), standardize))

######################################### variance explained by each regressor ###################
get_lmg_with_ci <- function(model, savepath, savepath_sink, nboot = 1000, level = 0.95) {
  
  # 1) Bootstrap
  set.seed(123)
  #be <- readRDS(file = savepath)

  boot <- boot.relimp(model, b = nboot, type = "lmg")
  be <- booteval.relimp(boot, level = level, bty = "perc", norank=T, nodiff = T)
  saveRDS(be, file = savepath) 
  
  # 2) Bootstrap mean LMG estimate
  boot_lmg_point <- be@lmg * 100

  # 3) CIs
  lower <- be$lmg.lower * 100
  upper <- be$lmg.upper * 100
  
  # 5) Format nicely for input to tibble later
  sprintf("%.1f<br>(%.1f-%.1f)", boot_lmg_point, lower, upper)
}

print('bootstrap rr')
rr_lm = lm(RR..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
                   data = df_scaled)
lmg_rr <- get_lmg_with_ci(rr_lm, './rr_be.rds', './rr_be.txt')

print('bootstrap rrv')
rrv_lm = lm(RRV..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
lmg_rrv <- get_lmg_with_ci(rrv_lm, './rrv_be.rds', './rrv_be.txt')

print('bootstrap rv')
rv_lm = lm(RV..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
lmg_rv <- get_lmg_with_ci(rv_lm, './rv_be.rds', './rv_be.txt')

print('bootstrap hr')
HR_lm = lm(HR..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
lmg_hr <- get_lmg_with_ci(HR_lm, './hr_be.rds', './hr_be.txt')

print('bootstrap hrv')
HRV_lm = lm(HRV..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
lmg_hrv <- get_lmg_with_ci(HRV_lm, './hrv_be.rds', './hrv_be.txt')

print('bootstrap pvi')
PVI_lm = lm(PVI..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
lmg_pvi <- get_lmg_with_ci(PVI_lm, './pvi_be.rds', './pvi_be.txt')

print('bootstrap spo2')
SPO2_lm = lm(SpO2..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
lmg_spo2 <- get_lmg_with_ci(SPO2_lm, './spo2_be.rds', './spo2_be.txt')

print('bootstrap fd')
FD_lm = lm(Mean.FD..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
lmg_fd <- get_lmg_with_ci(FD_lm, './fd_be.rds', './fd_be.txt')

regressors_phgy <-c('isoflurane', 'dexmedetomidine', 'session', 'strain:isoflurane', 'sex:isoflurane', "strain:dexmedetomidine", "sex:dexmedetomidine", 'strain', 'sex', 'time after isoflurane change')


relimp_tibble_phgy <-tibble(regressors_phgy, lmg_rr, lmg_rrv, lmg_rv, lmg_hr, lmg_hrv, lmg_pvi, lmg_spo2, lmg_fd) |>
  arrange(factor(regressors_phgy, levels = c('strain', 'sex', 'session', 'isoflurane', 'dexmedetomidine', 'time after isoflurane change', 'strain:isoflurane', 'sex:isoflurane', "strain:dexmedetomidine", "sex:dexmedetomidine"))) %>%
  add_row(regressors_phgy = 'subject', lmg_rr = '16', lmg_rrv = '11', lmg_rv = '24', lmg_hr = '22', lmg_hrv = '55', lmg_pvi = '47', lmg_spo2 = '16', lmg_fd = '18') %>%
  add_row(regressors_phgy = 'Total', lmg_rr = '83', lmg_rrv = '39', lmg_rv = '82', lmg_hr = '81', lmg_hrv = '76', lmg_pvi = '63', lmg_spo2 = '58', lmg_fd = '50') 

write.csv(relimp_tibble_phgy, "./phgy_r2_ci.csv")

relimp_table_phgy <- gt(relimp_tibble_phgy)|>
  fmt_markdown(columns = c(lmg_rr, lmg_rrv, lmg_rv, lmg_hr, lmg_hrv, lmg_pvi, lmg_spo2, lmg_fd)
  ) |>
  tab_header(
    title = md("**Table 2: Importance of regressors in predicting physiological metrics and motion**"),
    subtitle = "% of variance explained by each regressor"
  ) |>
  cols_label(
    regressors_phgy = md("**Regressors**"),
    lmg_rr = "RR", lmg_rrv = "RRV", lmg_rv = "RV", lmg_hr = "HR", lmg_hrv = "HRV", lmg_pvi = "PVI", lmg_spo2 = md("SpO2"), lmg_fd = "mean FD"
  )|> 
  cols_align(align = 'left') |>
  tab_options(column_labels.background.color = "darkgray")|>
  tab_source_note(
    source_note = md("The variance explained (R^2^) by a regressor is calculated as: the increase in model R^2^ when that regressor is added to the model. This value depends on the order in which that regressor is added to the model, thus the LMG metric represents the average R^2^ across all orderings, confidence intervals computed via bootstrapping are in brackets. Values were computed with the relaimpo package (Groemping, 2007). LMG values are not available for the random effect of subject, thus the R^2^ for subject was obtained by substracting the fixed effects R^2^ from the total effects R^2^. The most important regressor for each metric is highlighted in yellow.")
  )|>
  tab_style(
    style = cell_fill(color = "lightgray"),
    locations = cells_body(rows = lmg_rr>80)
  )|>
  tab_style(
    style = cell_fill(color = "lightyellow"),
    locations = cells_body(columns = lmg_rr, rows = c(1))
  )|>
  tab_style(
    style = cell_fill(color = "lightyellow"),
    locations = cells_body(columns = lmg_rrv, rows = c(1))
  )|>
  tab_style(
    style = cell_fill(color = "lightyellow"),
    locations = cells_body(columns = lmg_rv, rows = c(11))
  )|>
  tab_style(
    style = cell_fill(color = "lightyellow"),
    locations = cells_body(columns = lmg_hr, rows = c(1))
  )|>
  tab_style(
    style = cell_fill(color = "lightyellow"),
    locations = cells_body(columns = lmg_hrv, rows = c(11))
  )|>
  tab_style(
    style = cell_fill(color = "lightyellow"),
    locations = cells_body(columns = lmg_pvi, rows = c(11))
  )|>
  tab_style(
    style = cell_fill(color = "lightyellow"),
    locations = cells_body(columns = lmg_spo2, rows = c(3))
  )|>
  tab_style(
    style = cell_fill(color = "lightyellow"),
    locations = cells_body(columns = lmg_fd, rows = c(1))
  )|>
  tab_spanner(
    label = "LMG",
    columns = c(lmg_rr, lmg_rrv, lmg_rv, lmg_hr, lmg_hrv, lmg_pvi, lmg_spo2, lmg_fd)
  )
relimp_table_phgy
gtsave(relimp_table_phgy, "./phgy_r2_table_ci.png")