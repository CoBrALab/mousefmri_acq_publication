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

#the purpose of this script is to use brms to formally run the mediation analysis and generate the tables in the paper supplementary that
# indicate effect sizes and confidence intervals for both direct and indirect paths.

#load CAP and CPP outputs
DR_df=read.csv('../DR_analysis-to_extract_network_detectability/final_outputs/master_DR_and_phgy_variableWindow_sparse.csv')
#fisherz
DR_df$Average.correlation.to.network.fisherz <- psych::fisherz(DR_df$Average.correlation.to.network)

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

############################################### mediation analysis with brms ###################################
#need to do the mediation analysis formally with a package in order to get the proper significance (confidence intervals)

#define and run the model
set.seed(123)
f1 <- bf(RR..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID))
f2 <- bf(RRV..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID))
f5 <- bf(RV..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID))
f6 <- bf(HR..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID))
f7 <- bf(HRV..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID))
f8 <- bf(PVI..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc+ actual_ses_order + Time.after.isoflurane.change + (1|subject_ID))
f10 <- bf(SpO2..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID))
f11 <- bf(Mean.FD..mean.in.window ~  strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID))
f12 <- bf(network_detectability ~ RR..mean.in.window + RRV..mean.in.window + RV..mean.in.window + HR..mean.in.window + PVI..mean.in.window + HRV..mean.in.window + SpO2..mean.in.window + Mean.FD..mean.in.window + strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change + (1|subject_ID))
med <- brm(f1 + f2 + f5 + f6 + f7 + f8+ f10 + f11 + f12 + set_rescor(FALSE), data = df_scaled, refresh = 0)

#second model for path c and b - copy paste the outputs into a .txt
medc <- brm(network_detectability ~ strain + sex + isoflurane_percent +  + dex_conc + actual_ses_order + Time.after.isoflurane.change + strain:isoflurane_percent + isoflurane_percent:sex + strain:dex_conc + sex:dex_conc + (1|subject_ID), data = df_scaled, refresh = 0)
medb <- brm(network_detectability ~ RR..mean.in.window + RRV..mean.in.window + RV..mean.in.window + HR..mean.in.window + PVI..mean.in.window + HRV..mean.in.window + SpO2..mean.in.window + Mean.FD..mean.in.window + strain + sex + isoflurane_percent + dex_conc + actual_ses_order + Time.after.isoflurane.change + strain:isoflurane_percent + isoflurane_percent:sex + strain:dex_conc + sex:dex_conc + (1|subject_ID), data = df_scaled, refresh = 0)

#########################################################create tables of results
mediator_var_list <- list('RR..mean.in.window', 'RRV..mean.in.window', 'RV..mean.in.window', 'HR..mean.in.window', 'PVI..mean.in.window', 'HRV..mean.in.window', 'SpO2..mean.in.window', 'Mean.FD..mean.in.window')
mediator_var_list2 <- list('RRmeaninwindow',  'RRVmeaninwindow', 'RVmeaninwindow', 'HRmeaninwindow', 'PVImeaninwindow', 'HRVmeaninwindow', 'SpO2meaninwindow', 'MeanFDmeaninwindow')
mediator_var_list3 <- list('RR',  'RRV', 'RV', 'HR', 'PVI', 'HRV', 'SpO2', 'MeanFD')
indep_var_list <- list('strainC3HeBDFeJ', 'sexf', 'isoflurane_percent2M1',  'isoflurane_percent3M2','dex_conc2M1', 'dex_conc3M2', 'actual_ses_order2M1', 'actual_ses_order3M2', 'Time.after.isoflurane.change')
indep_var_list2 <- list('strain (C3HeB/FeJ)', 'sex (f)', 'isoflurane (0.5% vs 0.23%)',  'isoflurane (1% vs 0.5%)','dexmedetomidine (0.05 vs 0.025)', 'dexmedetomidine (0.1 vs 0.05)',  'session (2 vs 1)', 'session (3 vs 2)','Time after isoflurane change')
indep_var_list_full <- list('strainC3HeBDFeJ', 'sexf', 'isoflurane_percent2M1',  'isoflurane_percent3M2','dex_conc2M1', 'dex_conc3M2', 'actual_ses_order2M1', 'actual_ses_order3M2', 'Time.after.isoflurane.change', 'strainC3HeBDFeJ:isoflurane_percent2M1', 'strainC3HeBDFeJ:isoflurane_percent3M2', 'isoflurane_percent2M1:sexf', 'isoflurane_percent3M2:sexf', 'strainC3HeBDFeJ:dex_conc2M1', 'strainC3HeBDFeJ:dex_conc3M2', 'sexf:dex_conc2M1', 'sexf:dex_conc3M2')
indep_var_list2_full <- list('strain (C3HeB/FeJ)', 'sex (f)', 'isoflurane (0.5% vs 0.23%)',  'isoflurane (1% vs 0.5%)','dexmedetomidine (0.05 vs 0.025)', 'dexmedetomidine (0.1 vs 0.05)', 'session (2 vs 1)', 'session (3 vs 2)', 'Time after isoflurane change', 'strain : isoflurane (0.5% vs 0.23%)', 'strain : isoflurane (1% vs 0.5%)', 'sex : isoflurane (0.5% vs 0.23%)', 'sex : isoflurane (1% vs 0.5%)', 'strain : dexmedetomdine (0.05 vs 0.025)', 'strain : dexmedetomidine (0.1 vs 0.05)', 'sex : dexmedetomidine (0.05 vs 0.025)', 'sex : dexmedetomidine (0.1 vs 0.05)')

#print all the direct effects
hypothesis(med, "networkdetectability_sexf  = 0")
hypothesis(med, "networkdetectability_strainC3HeBDFeJ  = 0")
hypothesis(med, "networkdetectability_isoflurane_percent2M1  = 0")
hypothesis(med, "networkdetectability_isoflurane_percent3M2  = 0")
hypothesis(med, "networkdetectability_actual_ses_order2M1  = 0")
hypothesis(med, "networkdetectability_actual_ses_order3M2  = 0")
hypothesis(med, "networkdetectability_dex_conc2M1  = 0")
hypothesis(med, "networkdetectability_dex_conc3M2  = 0")
hypothesis(med, "networkdetectability_Time.after.isoflurane.change  = 0")

##################################path a

#print path a effects
iter = 1 #count the mediators
iter_i = 0 #count the indep variables
for (m in mediator_var_list) {
  for (i in indep_var_list) {
    i_on_m <- paste(mediator_var_list2[[iter]], i, sep = '_')
    expression <- paste(i_on_m,' = 0', sep = ' ')
    patha_effect <-hypothesis(med, expression)
    print(patha_effect$hypothesis)
    if ((iter == 1) & (iter_i == 0)) {
      patha_effects_tibble<-tibble(mediator_var_list3[[iter]], indep_var_list2[[(iter_i%%9)+1]], patha_effect$hypothesis)
    }
    else {
      #add a row if we're not on the first
      patha_effects_tibble <- patha_effects_tibble |>
        add_row(mediator_var_list3[[iter]], indep_var_list2[[(iter_i%%9)+1]], patha_effect$hypothesis)
    }
    iter_i = iter_i +1
  }
  iter = iter + 1
}

write.csv(patha_effects_tibble, "./mediation_stats_tables/patha_effects.csv")

#format as a table, selecting only the rows of interest
patha_effects_tibble <- read.csv('./mediation_stats_tables/patha_effects.csv')
summary(patha_effects_tibble)

patha_effects_table <- gt(patha_effects_tibble %>% 
                            dplyr::select(`mediator_var_list3..iter..`, `indep_var_list2...iter_i..9....1..`, Estimate, CI.Lower, CI.Upper, Star)%>%
                            dplyr::mutate(across(where(is.numeric), round, 3)))%>%
  tab_header(
    title = md("Table S1: Dependence of mediators (physiology, motion) on independent variables (demographics, anesthesia, session, time)"),
    subtitle = "Path A of mediation analysis"
  ) %>%
  cols_label(
    `mediator_var_list3..iter..` = md("**Mediator Variable**"),
    `indep_var_list2...iter_i..9....1..` = md("**Independent Variable**"),
    Estimate = md("**Effect estimate (standardized)**"),
    CI.Lower = md("**Confidence Interval (lower)**"),
    CI.Upper = md("**Confidence Interval (upper)**"),
    Star = md("**Significant?**")
  )%>%
  cols_align(align = 'center') %>%
  tab_options(column_labels.background.color = "darkgray") %>%
  tab_style(
    style = cell_fill(color = "lightgray"),
    locations = cells_body(rows = mediator_var_list3..iter.. == 'RRV')) %>%
  tab_style(
    style = cell_fill(color = "lightgray"),
    locations = cells_body(rows = mediator_var_list3..iter.. == 'HR')) %>%
  tab_style(
    style = cell_fill(color = "lightgray"),
    locations = cells_body(rows = mediator_var_list3..iter.. == 'HRV')) %>%
  tab_style(
    style = cell_fill(color = "lightgray"),
    locations = cells_body(rows = mediator_var_list3..iter.. == 'MeanFD')) %>%
  tab_source_note(
    source_note = md("The effect of independent variables on mediator variables, modeled as a linear mixed effects regression using Bayesian statistics. All mediator variables are continuous and standardized to a mean of 0 and standard deviation of 1. Thus the effect estimate indicates the mean difference in the mediator variable (eg RR) when comparing the two contrasts (e.g. two strains), expressed in units of standard deviation. The 95% confidence intervals indicate the lowest and highest esimates for the effect size. If the confidence interval crosses 0, then the effect is not significant."))
patha_effects_table

gtsave(patha_effects_table, "./mediation_stats_tables/patha_effects_table.png")

######################### path c (total)
pathc_effects_tibble <- read.table('./mediation_stats_tables/pathctotal_outputs_noquadratic.txt', header = TRUE, stringsAsFactors = FALSE) %>%
  mutate(Star = case_when(
    lowerCI < 0 & upperCI <=0 ~ '*',
    lowerCI >=0 & upperCI > 0 ~ '*',
    lowerCI <=0 & upperCI >= 0 ~ ''
  )) %>%
  slice(-1)

pathc_effects_tibble$Predictors <- indep_var_list2_full


pathc_effects_table <- gt(pathc_effects_tibble %>% 
                            dplyr::select(`Predictors`, Estimate, lowerCI, upperCI, Star)%>%
                            dplyr::mutate(across(where(is.numeric), round, 3)))%>%
  tab_header(
    title = md("Table S2: Dependence of network detectability on independent variables"),
    subtitle = "Path C of the mediation analysis"
  ) %>%
  cols_label(
    `Predictors` = md("**Independent Variable**"),
    Estimate = md("**Effect estimate**"),
    lowerCI = md("**Confidence Interval (lower)**"),
    upperCI = md("**Confidence Interval (upper)**"),
    Star = md("**Significant?**")
  )%>%
  cols_align(align = 'center') %>%
  tab_options(column_labels.background.color = "darkgray")%>%
  #gt_split(row_every_n = 22) %>%
  tab_source_note(
    source_note = md("The total effects of independent variables on network detectability, modeled as a linear mixed effects regression using Bayesian statistics. Network detectability is a spatial correlation that was Fisher-Z and log transformed. Thus the effect estimate indicates the mean difference in network detectability when comparing the two contrasts (e.g. two strains), expressed in units of log correlations. The 95% confidence intervals indicate the lowest and highest esimates for the effect size. If the confidence interval crosses 0, then the effect is not significant."))

pathc_effects_table
gtsave(pathc_effects_table, "./mediation_stats_tables/pathctotal_effects_table.png")


######################### path b
pathb_effects_tibble <- read.table('./mediation_stats_tables/pathb_outputs_noquadratic.txt', header = TRUE, stringsAsFactors = FALSE) %>%
  mutate(Star = case_when(
    lowerCI < 0 & upperCI <0 ~ '*',
    lowerCI >=0 & upperCI > 0 ~ '*',
    lowerCI <=0 & upperCI >= 0 ~ ''
  )) %>%
  slice(-1)
pathb_effects_tibble$Predictors <- mediator_var_list3


pathb_effects_table <- gt(pathb_effects_tibble %>% 
                            dplyr::select(`Predictors`, Estimate, lowerCI, upperCI, Star)%>%
                            dplyr::mutate(across(where(is.numeric), round, 3)))%>%
  tab_header(
    title = md("Table S4: Mediator (instantaneous) variables that are predictive of network detectability above and beyond the independent variables."),
    subtitle = "Path B of the mediation analysis"
  ) %>%
  cols_label(
    `Predictors` = md("**Mediator Variable**"),
    Estimate = md("**Effect estimate**"),
    lowerCI = md("**Confidence Interval (lower)**"),
    upperCI = md("**Confidence Interval (upper)**"),
    Star = md("**Significant?**")
  )%>%
  cols_align(align = 'center') %>%
  tab_options(column_labels.background.color = "darkgray")%>%
  #gt_split(row_every_n = 22) %>%
  tab_source_note(
    source_note = md("The effects of mediator variables on network detectability when controlling for independent variables, modeled as a linear mixed effects regression using Bayesian statistics. Network detectability is a spatial correlation that was Fisher-Z and log transformed. The 95% confidence intervals indicate the lowest and highest esimates for the effect size. If the confidence interval crosses 0, then the effect is not significant."))

pathb_effects_table
gtsave(pathb_effects_table, "./mediation_stats_tables/pathb_effects_table.png")

############################ path c' (direct effect when controlling for mediators)
#print path c effects
iter_i = 1
for (i in indep_var_list_full) {
  i_on_y <- paste('networkdetectability', i, sep = '_')
  expression <- paste(i_on_y, ' = 0', sep = ' ')
  pathc_effect <- hypothesis(med, expression)
  print(pathc_effect$hypothesis)
  if (iter_i == 1){
    pathcprime_effects_tibble<-tibble(indep_var_list2_full[[iter_i]], pathc_effect$hypothesis)
  }
  else {
    #add a row if we're not on the first
    pathcprime_effects_tibble <- pathcprime_effects_tibble |>
      add_row(indep_var_list2_full[[iter_i]], pathc_effect$hypothesis)
  }
  iter_i = iter_i + 1
}
write.csv(pathcprime_effects_tibble, "./mediation_stats_tables/pathcprime_effects.csv")

#format as a table, selecting only the rows of interest
pathcprime_effects_tibble <- read.csv('./mediation_stats_tables/pathcprime_effects.csv')
pathcprime_effects_table <- gt(pathcprime_effects_tibble %>% 
                                 dplyr::select(`indep_var_list2_full..iter_i..`, Estimate, CI.Lower, CI.Upper, Star)%>%
                                 dplyr::mutate(across(where(is.numeric), round, 3)))%>%
  tab_header(
    title = md("Table S3: Dependence of network detectability on independent variables while controlling for mediator variables"),
    subtitle = "Path C' of the mediation analysis"
  ) %>%
  cols_label(
    `indep_var_list2_full..iter_i..` = md("**Independent Variable**"),
    Estimate = md("**Effect estimate**"),
    CI.Lower = md("**Confidence Interval (lower)**"),
    CI.Upper = md("**Confidence Interval (upper)**"),
    Star = md("**Significant?**")
  )%>%
  cols_align(align = 'center') %>%
  tab_options(column_labels.background.color = "darkgray")%>%
  #gt_split(row_every_n = 22) %>%
  tab_source_note(
    source_note = md("The direct effects of independent variables on network detectability obtained when controlling for all mediator variables, modeled as a linear mixed effects regression using Bayesian statistics. Network detectability is a spatial correlation that was Fisher-Z and log transformed. Thus the effect estimate indicates the mean difference in network detectability when comparing the two contrasts (e.g. two strains), expressed in units of log correlations. The 95% confidence intervals indicate the lowest and highest esimates for the effect size. If the confidence interval crosses 0, then the effect is not significant."))

pathcprime_effects_table
gtsave(pathcprime_effects_table, "./mediation_stats_tables/pathcprime_effects_table.png")

############################################ indirect effects
#print all the indirect effects in a loop - interaction terms are not shown because cannot be interpreted as mediators                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
df_mediation_estimates <- data.frame(matrix(nrow = 0, ncol=8))
colnames(df_mediation_estimates) = c("Hypothesis", "Estimate", "Est.Error",  "CI.Lower", "CI.Upper", "Evid.Ratio", "Post.Prob", "Star")
iter = 1 #count the mediators
iter_i = 0 #count the indep variables
for (m in mediator_var_list) {
  for (i in indep_var_list) {
    i_on_m <- paste(mediator_var_list2[[iter]], i, sep = '_')
    m_on_y <- paste('*networkdetectability', m, sep = '_')
    expression <- paste(i_on_m, m_on_y, ' = 0', sep = ' ')
    print(mediator_var_list3[[iter]])
    print(indep_var_list2[[(iter_i%%9)+1]])
    print(expression)
    indirect_effect <- hypothesis(med, expression)
    print(indirect_effect$hypothesis)
    if ((iter == 1) & (iter_i == 0)) {
      #initialize the tibble if we're on the first mediator and indep var
      med_indirect_effects_tibble<-tibble(mediator_var_list3[[iter]], indep_var_list2[[(iter_i%%9)+1]], indirect_effect$hypothesis)
    }
    else {
      #add a row if we're not on the first
      med_indirect_effects_tibble <- med_indirect_effects_tibble |>
        add_row(mediator_var_list3[[iter]], indep_var_list2[[(iter_i%%9)+1]], indirect_effect$hypothesis)
    }
    iter_i = iter_i +1
  }
  iter = iter + 1
}
write.csv(med_indirect_effects_tibble, "./mediation_stats_tables/indirect_effects.csv")
summary(med_indirect_effects_tibble)

#format as a table, selecting only the rows of interest
med_indirect_effects_tibble <- read.csv('./mediation_stats_tables/indirect_effects.csv')
indirect_effects_table <- gt(med_indirect_effects_tibble %>% 
                               dplyr::select(`mediator_var_list3..iter..`, `indep_var_list2...iter_i..9....1..`, Estimate, CI.Lower, CI.Upper, Star) %>%
                               dplyr::mutate(across(where(is.numeric), round, 3))) %>%
  tab_header(
    title = md("Table S5: Identifying mediators between independent variables and network detectability"),
    subtitle = "Path C - Path C' (i.e. total-direct effects)"
  ) %>%
  cols_label(
    `mediator_var_list3..iter..` = md("**Mediator Variable**"),
    `indep_var_list2...iter_i..9....1..` = md("**Independent Variable**"),
    Estimate = md("**Effect estimate**"),
    CI.Lower = md("**Confidence Interval (lower)**"),
    CI.Upper = md("**Confidence Interval (upper)**"),
    Star = md("**Significant?**")
  )%>%
  cols_align(align = 'center') %>%
  #gt_split(row_every_n = 22) %>%
  tab_options(column_labels.background.color = "darkgray")%>%
  tab_style(
    style = cell_fill(color = "lightgray"),
    locations = cells_body(rows = mediator_var_list3..iter.. == 'RRV')) %>%
  tab_style(
    style = cell_fill(color = "lightgray"),
    locations = cells_body(rows = mediator_var_list3..iter.. == 'HR')) %>%
  tab_style(
    style = cell_fill(color = "lightgray"),
    locations = cells_body(rows = mediator_var_list3..iter.. == 'HRV')) %>%
  tab_style(
    style = cell_fill(color = "lightgray"),
    locations = cells_body(rows = mediator_var_list3..iter.. == 'MeanFD')) %>%
  tab_source_note(
    source_note = md("The determination of which mediator variables are significant mediators of the relationship between independent variables and network detectability. This indicates that the independent variables have an indirect effect on network detectability through these mediator variables. The effect sizes are the difference between the total and direct effects. 95% confidence intervals indicate the lowest and highest esimates for the effect size. If the confidence interval crosses 0, then the effect is not significant."))

indirect_effects_table
gtsave(indirect_effects_table, "./mediation_stats_tables/indirect_effects_table.png")

