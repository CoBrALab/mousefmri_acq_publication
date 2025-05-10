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

#load CAP and CPP outputs
DR_df=read.csv('../DR_analysis-to_extract_network_detectability/final_outputs/master_DR_and_phgy_variableWindow_sparse.csv')

#fisherz
DR_df$Average.correlation.to.network.fisherz <- psych::fisherz(DR_df$Average.correlation.to.network)
write.csv(DR_df, "./mediation_path_effectSize_csvs/master_DR_and_phgy_variableWindow_sparse_fisherzND.csv")

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

############################################### RUN LMER FOR EACH PATH ##############################################################
# this is essentially an informal mediation analysis, we check the effect of the indep var with and without controlling for physiology. 
#these outputs are used to obtain effect size estimates for plotting purposes only (since the effects package won't work on the mediation outputs).
#Later we do a more formal mediation analysis with the brms package to properly estimate significance .

#remove quadratic var because confusing everything - Mean.FD..std.in.window were removed because collinear. Note PVI is highly correlated to HRV (mean in window)
total_path = lmer(network_detectability ~  RR..mean.in.window + RRV..mean.in.window + RV..mean.in.window + HR..mean.in.window +PVI..mean.in.window + HRV..mean.in.window + SpO2..mean.in.window + Mean.FD..mean.in.window + strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID),
                  data = df_scaled)
total_path = lm(network_detectability ~  RR..mean.in.window + RRV..mean.in.window + RV..mean.in.window + HR..mean.in.window +PVI..mean.in.window + HRV..mean.in.window + SpO2..mean.in.window + Mean.FD..mean.in.window,
                  data = df_scaled)
check_model(total_path) #the collinearity here is between main effects and interaction terms, this is normal
summary(total_path)
summ(total_path) #, 53%, 64%

#now see if any variables have more significance without the predictors
pathc = lmer(network_detectability ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID),
             data = df_scaled) #I'm excluding age_days because highly collinear with actual_ses_order
check_model(pathc)
summary(pathc)
summ(pathc) #0.44 explained by fixed, 0.61 total


#now see if accounting for dex*weigh explains the interactions between strain and sex (ie a dosing effect) - no it doesn't, it's also sig but the main effects of sex*dex don't go away
total_path2 = lmer(network_detectability ~  RR..mean.in.window+ RRV..mean.in.window + RV..mean.in.window + HR..mean.in.window +PVI..mean.in.window + HRV..mean.in.window + SpO2..mean.in.window + Mean.FD..mean.in.window + strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + weight*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID),
                  data = df_scaled)
check_model(total_path2) #the collinearity here is between main effects and interaction terms, this is normal
summary(total_path2)


patha0 = lmer(RR..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df_scaled)
patha1 = lmer(RRV..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc+ actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df_scaled)
patha2 = lmer(RV..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df_scaled)
patha3 = lmer(HR..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc+ actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df_scaled)
patha4 = lmer(HRV..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df_scaled)
patha5 = lmer(PVI..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df_scaled)
patha6 = lmer(SpO2..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df_scaled)
patha7 = lmer(Mean.FD..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df_scaled)
summary(patha0)

#non zscored version
patha0 = lmer(RR..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df)
patha1 = lmer(RRV..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc+ actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df)
patha2 = lmer(RV..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df)
patha3 = lmer(HR..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc+ actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df)
patha4 = lmer(HRV..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df)
patha5 = lmer(PVI..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df)
patha6 = lmer(SpO2..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df)
patha7 = lmer(Mean.FD..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID), data = df)
summary(patha0)
summary(patha1)
summary(patha2)
summary(patha3)
summary(patha4)
summary(patha5)
summary(patha6)
summary(patha7)
#all phgy var are probably mediators except PVI. They all impact network detectability in part because they are reflective of changes in iso, ses and weight.

#save marginal effects for pathc
fitdata_strain_sex = as.data.frame(Effect(c("strain", "sex"), pathc))
fitdata_iso = as.data.frame(Effect(c("isoflurane_percent"), pathc))
fitdata_ses= as.data.frame(Effect(c("actual_ses_order"), pathc))
fitdata_dex = as.data.frame(Effect(c("dex_conc"), pathc))
fitdata_time = as.data.frame(Effect(c("Time.after.isoflurane.change"), pathc, xlevels=list(Time.after.isoflurane.change=seq(-1.3285,1.3799,0.9028))))
write.csv(fitdata_strain_sex, "./mediation_path_effectSize_csvs/pathc_effects_nd_strain_sex.csv")
write.csv(fitdata_iso, "./mediation_path_effectSize_csvs/pathc_effects_nd_iso.csv")
write.csv(fitdata_ses, "./mediation_path_effectSize_csvs/pathc_effects_nd_ses.csv")
write.csv(fitdata_time, "./mediation_path_effectSize_csvs/pathc_effects_nd_time.csv")
write.csv(fitdata_dex, "./mediation_path_effectSize_csvs/pathc_effects_nd_dex.csv")

#save marginal effects for patha
patha_list = list(patha0, patha1, patha2, patha3, patha4, patha5, patha6, patha7)
med_var_list = list('RR', 'RRV', 'RV', 'HR', 'HRV', 'PVI', 'SpO2', 'Mean FD')
i=1
for (patha in patha_list) {
  fitdata_strain_sex = as.data.frame(Effect(c("strain", "sex"), patha))
  fitdata_iso = as.data.frame(Effect(c("isoflurane_percent"), patha))
  fitdata_ses= as.data.frame(Effect(c("actual_ses_order"), patha))
  fitdata_dex = as.data.frame(Effect(c("dex_conc"), patha))
  fitdata_time = as.data.frame(Effect(c("Time.after.isoflurane.change"), patha,  xlevels=list(Time.after.isoflurane.change=seq(-1.3285,1.3799,0.9028))))
  basepath = "./mediation_path_effectSize_csvs/patha"
  write.csv(fitdata_strain_sex, paste(basepath, med_var_list[[i]], 'strain_sex.csv', sep = '_'))
  write.csv(fitdata_iso, paste(basepath, med_var_list[[i]], 'iso.csv', sep = '_'))
  write.csv(fitdata_ses, paste(basepath, med_var_list[[i]], 'ses.csv', sep = '_'))
  write.csv(fitdata_dex, paste(basepath, med_var_list[[i]], 'dex.csv', sep = '_'))
  write.csv(fitdata_time, paste(basepath, med_var_list[[i]], 'time.csv', sep = '_'))
  i = i+1
}

# save the marginal effects for path b
fitdata_RR = as.data.frame(Effect(c('RR..mean.in.window'), total_path, xlevels=list(RR..mean.in.window=seq(-2,3,0.02))))
write.csv(fitdata_RR, "./mediation_path_effectSize_csvs/pathb_nd_RR.csv")

fitdata_RV = as.data.frame(Effect(c("RV..mean.in.window"), total_path, xlevels=list(RV..mean.in.window=seq(-2,3,0.02))))
write.csv(fitdata_RV, "./mediation_path_effectSize_csvs/pathb_nd_RV.csv")

fitdata_RRV = as.data.frame(Effect(c("RRV..mean.in.window"), total_path, xlevels=list(RRV..mean.in.window=seq(-2,8.5,0.02))))
write.csv(fitdata_RRV, "./mediation_path_effectSize_csvs/pathb_nd_RRV.csv")

fitdata_HR = as.data.frame(Effect(c("HR..mean.in.window"), total_path, xlevels=list(HR..mean.in.window=seq(-2,3.5,0.02))))
write.csv(fitdata_HR, "./mediation_path_effectSize_csvs/pathb_nd_HR.csv")

fitdata_PVI = as.data.frame(Effect(c("PVI..mean.in.window"), total_path, xlevels=list(PVI..mean.in.window=seq(-2.5,1.5,0.02))))
write.csv(fitdata_PVI, "./mediation_path_effectSize_csvs/pathb_nd_PVI.csv")

fitdata_HRV = as.data.frame(Effect(c("HRV..mean.in.window"), total_path, xlevels=list(HRV..mean.in.window=seq(-2,3.5,0.02))))
write.csv(fitdata_HRV, "./mediation_path_effectSize_csvs/pathb_nd_HRV.csv")

fitdata_SpO2 = as.data.frame(Effect(c("SpO2..mean.in.window"), total_path, xlevels=list(SpO2..mean.in.window=seq(-4,1.5,0.02))))
write.csv(fitdata_SpO2, "./mediation_path_effectSize_csvs/pathb_nd_SpO2.csv")

fitdata_Mean.FD = as.data.frame(Effect(c("Mean.FD..mean.in.window"), total_path, xlevels=list(Mean.FD..mean.in.window=seq(-2,7,0.02))))
write.csv(fitdata_Mean.FD, "./mediation_path_effectSize_csvs/pathb_nd_Mean FD.csv")



######################################### variance explained by each regressor ###################
#calculate variables importance metrics for the basic model (no interactions because first and last metrics can't be calculated)
total_path_lm = lm(network_detectability ~  RR..mean.in.window + RRV..mean.in.window + RV..mean.in.window + HR..mean.in.window +PVI..mean.in.window + HRV..mean.in.window + SpO2..mean.in.window + Mean.FD..mean.in.window + strain + isoflurane_percent + sex +dex_conc + actual_ses_order + Time.after.isoflurane.change,
                  data = df_scaled)
relimp <-calc.relimp(total_path_lm, type = c('lmg', 'last', 'first'), rela = FALSE)

pathc_lm = lm(network_detectability ~   strain + isoflurane_percent + sex +dex_conc + actual_ses_order + Time.after.isoflurane.change,
                   data = df_scaled)
relimp_c <-calc.relimp(pathc_lm, type = c('lmg'), rela = FALSE)

#redo but with interaction terms
total_path_lm_interac = lm(network_detectability ~  RR..mean.in.window + RRV..mean.in.window + RV..mean.in.window + HR..mean.in.window +PVI..mean.in.window + HRV..mean.in.window + SpO2..mean.in.window + Mean.FD..mean.in.window + strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
                           data = df_scaled)
relimp_interac <-calc.relimp(total_path_lm_interac, type = c('lmg'), rela = FALSE)


regressors <-c('isoflurane', 'dexmedetomidine', 'session', 'RR', 'RRV', 'RV', 'HR', 'PVI', 'HRV', 'SpO2', 'mean FD', 'strain', 'sex', 'time after isoflurane change')
LMG <-round(relimp@lmg*100,1)
first<-round(relimp@first*100,1)
last<-round(relimp@last*100,1)
relimp_tibble <-tibble(regressors, LMG, first, last) |>
  arrange(factor(regressors, levels = c('strain', 'sex', 'session', 'isoflurane', 'dexmedetomidine', 'time after isoflurane change', "RR", "RRV", "RV", "HR", "HRV", "PVI", "SpO2", "mean FD"))) %>%
  add_row(regressors = 'total instantaneous', LMG = 14.7, first = 23.4, last = 4.6) 
  
relimp_table <- gt(relimp_tibble)|>
  tab_header(
    title = md("**Table 2: Importance of regressors in predicting network detectability**"),
    subtitle = "% of variance in network detectability explained by each regressor,\n computed in 3 ways"
  ) |>
  cols_label(
    regressors = md("**Regressors**"),
    LMG = md("**LMG**"),
    first = md("**First**"),
    last = md("**Last**")
  )|> 
  cols_align(align = 'left') |>
  tab_options(column_labels.background.color = "darkgray")|>
  tab_source_note(
    source_note = md("The variance explained (R^2^) by a regressor is calculated as: the increase in model R^2^ when that regressor is added to the model. This value depends on the order in which that regressor is added to the model, thus we present 3 approachs. LMG is the average R^2^ across all orderings. 'First' is when that regressor is added first. 'Last' is when that regressor is added last. Values were computed with the relaimpo package (Groemping, 2007). Interaction terms and random effects are not shown as their 'first' and 'last' contribution cannot be calculated with the package.")
    )|>
  tab_row_group(
    label = "independent variables",
    rows = 1:6
  )|>
  tab_row_group(
    label = "instantaneous variables",
    rows = 7:15
  ) |>
  tab_style(
    style = cell_fill(color = "lightyellow"),
    locations = cells_body(rows = c(4))
  ) |>
  tab_style(
    style = cell_fill(color = "lightyellow"),
    locations = cells_body(columns = first, rows = c(15))
  ) |>
  row_group_order(groups = c('independent variables', "instantaneous variables")
                  )|>
  tab_style(
    style = cell_fill(color = "lightgray"),
    locations = cells_row_groups(groups = 1)
  )|>
  tab_style(
    style = cell_fill(color = "lightgray"),
    locations = cells_row_groups(groups = 2)
  )
relimp_table
gtsave(relimp_table, "./variance_explained_tables/nd_r2_table.png")


#I could do bootstrapping to get confidence intervals, but the results are too messy
bootresults<-boot.relimp(total_path_lm_interac, b=1000, sort = TRUE) 
ci<-booteval.relimp(bootresults, norank=T)
ci
plot(ci)

rr_lm = lm(RR..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
                   data = df_scaled)
rr_relimp <-calc.relimp(rr_lm)
rrv_lm = lm(RRV..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
rrv_relimp <-calc.relimp(rrv_lm)
rv_lm = lm(RV..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
rv_relimp <-calc.relimp(rv_lm)
HR_lm = lm(HR..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
HR_relimp <-calc.relimp(HR_lm)
HRV_lm = lm(HRV..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
HRV_relimp <-calc.relimp(HRV_lm)
PVI_lm = lm(PVI..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
PVI_relimp <-calc.relimp(PVI_lm)
SPO2_lm = lm(SpO2..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
SPO2_relimp <-calc.relimp(SPO2_lm)
FD_lm = lm(Mean.FD..mean.in.window ~ strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
           data = df_scaled)
FD_relimp <-calc.relimp(FD_lm)

regressors_phgy <-c('isoflurane', 'dexmedetomidine', 'session', 'strain:isoflurane', 'sex:isoflurane', "strain:dexmedetomidine", "sex:dexmedetomidine", 'strain', 'sex', 'time after isoflurane change')
lmg_rr = round(rr_relimp@lmg*100,1) 
lmg_rrv = round(rrv_relimp@lmg*100,1)
lmg_rv = round(rv_relimp@lmg*100,1)
lmg_hr = round(HR_relimp@lmg*100,1)
lmg_hrv = round(HRV_relimp@lmg*100,1)
lmg_pvi = round(PVI_relimp@lmg*100,1)
lmg_spo2 = round(SPO2_relimp@lmg*100,1)
lmg_fd = round(FD_relimp@lmg*100,1)

print(lmg_rr)
print(lmg_rrv)
print(lmg_rv)
print(lmg_hr)
print(lmg_hrv)
print(lmg_pvi)
print(lmg_spo2)
print(lmg_fd)

relimp_tibble_phgy <-tibble(regressors_phgy, lmg_rr, lmg_rrv, lmg_rv, lmg_hr, lmg_hrv, lmg_pvi, lmg_spo2, lmg_fd) |>
  arrange(factor(regressors_phgy, levels = c('strain', 'sex', 'session', 'isoflurane', 'dexmedetomidine', 'time after isoflurane change', 'strain:isoflurane', 'sex:isoflurane', "strain:dexmedetomidine", "sex:dexmedetomidine"))) %>%
  add_row(regressors_phgy = 'subject', lmg_rr = 16, lmg_rrv = 11, lmg_rv = 24, lmg_hr = 22, lmg_hrv = 55, lmg_pvi = 47, lmg_spo2 = 16, lmg_fd = 18) %>%
  add_row(regressors_phgy = 'Total', lmg_rr = 83, lmg_rrv = 39, lmg_rv = 82, lmg_hr = 81, lmg_hrv = 76, lmg_pvi = 63, lmg_spo2 = 58, lmg_fd = 50) 

write.csv(relimp_tibble_phgy, "./variance_explained_tables/phgy_r2.csv")

relimp_table_phgy <- gt(relimp_tibble_phgy)|>
  tab_header(
    title = md("**Table 1: Importance of regressors in predicting physiological metrics and motion**"),
    subtitle = "% of variance explained by each regressor"
  ) |>
  cols_label(
    regressors_phgy = md("**Regressors**"),
    lmg_rr = "RR", lmg_rrv = "RRV", lmg_rv = "RV", lmg_hr = "HR", lmg_hrv = "HRV", lmg_pvi = "PVI", lmg_spo2 = md("SpO2"), lmg_fd = "mean FD"
  )|> 
  cols_align(align = 'left') |>
  tab_options(column_labels.background.color = "darkgray")|>
  tab_source_note(
    source_note = md("The variance explained (R^2^) by a regressor is calculated as: the increase in model R^2^ when that regressor is added to the model. This value depends on the order in which that regressor is added to the model, thus the LMG metric represents the average R^2^ across all orderings. Values were computed with the relaimpo package (Groemping, 2007). LMG values are not available for the random effect of subject, thus the R^2^ for subject was obtained by substracting the fixed effects R^2^ from the total effects R^2^. The most important regressor for each metric is highlighted in yellow.")
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
gtsave(relimp_table_phgy, "./variance_explained_tables/phgy_r2_table.png")


##########################################SEX STRAIN ZSCORED MEDIATION ########################3
#rerun total path using the sexstrain zscored phgy var, so that the lines are in the right space
total_path_sexstrain = lmer(network_detectability ~  RR.sexstrain_zscore..mean.in.window + RRV.sexstrain_zscore..mean.in.window + RV.sexstrain_zscore..mean.in.window + HR.sexstrain_zscore..mean.in.window +PVI.sexstrain_zscore..mean.in.window + HRV.sexstrain_zscore..mean.in.window + SpO2.sexstrain_zscore..mean.in.window + Mean.FD..mean.in.window + strain*isoflurane_percent + sex*isoflurane_percent + sex*dex_conc + strain*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID),
                            data = df_scaled)
summary(total_path_sexstrain)
fitdata_SpO2 = as.data.frame(Effect(c("SpO2.sexstrain_zscore..mean.in.window"), total_path_sexstrain, xlevels=list(SpO2.sexstrain_zscore..mean.in.window=seq(-10,2.5,0.02))))
write.csv(fitdata_SpO2, "./mediation_path_effectSize_csvs/pathb_nd_SpO2_sexstrain.csv")

fitdata_HRV = as.data.frame(Effect(c("HRV.sexstrain_zscore..mean.in.window"), total_path_sexstrain, xlevels=list(HRV.sexstrain_zscore..mean.in.window=seq(-1.5,3,0.02))))
write.csv(fitdata_HRV, "./mediation_path_effectSize_csvs/pathb_nd_HRV_sexstrain.csv")

fitdata_PVI = as.data.frame(Effect(c("PVI.sexstrain_zscore..mean.in.window"), total_path_sexstrain, xlevels=list(PVI.sexstrain_zscore..mean.in.window=seq(-3.5,1.5,0.02))))
write.csv(fitdata_PVI, "./mediation_path_effectSize_csvs/pathb_nd_PVI_sexstrain.csv")

fitdata_HR = as.data.frame(Effect(c("HR.sexstrain_zscore..mean.in.window"), total_path_sexstrain, xlevels=list(HR.sexstrain_zscore..mean.in.window=seq(-2.5,3,0.02))))
write.csv(fitdata_HR, "./mediation_path_effectSize_csvs/pathb_nd_HR_sexstrain.csv")

fitdata_RRV = as.data.frame(Effect(c("RRV.sexstrain_zscore..mean.in.window"), total_path_sexstrain, xlevels=list(RRV.sexstrain_zscore..mean.in.window=seq(-1,4,0.02))))
write.csv(fitdata_RRV, "./mediation_path_effectSize_csvs/pathb_nd_RRV_sexstrain.csv")

fitdata_RV = as.data.frame(Effect(c("RV.sexstrain_zscore..mean.in.window"), total_path_sexstrain, xlevels=list(RV.sexstrain_zscore..mean.in.window=seq(-2.5,3,0.02))))
write.csv(fitdata_RV, "./mediation_path_effectSize_csvs/pathb_nd_RV_sexstrain.csv")

fitdata_RR = as.data.frame(Effect(c('RR.sexstrain_zscore..mean.in.window'), total_path_sexstrain, xlevels=list(RR.sexstrain_zscore..mean.in.window=seq(-2,6,0.02))))
write.csv(fitdata_RR, "./mediation_path_effectSize_csvs/pathb_nd_RR_sexstrain.csv")
