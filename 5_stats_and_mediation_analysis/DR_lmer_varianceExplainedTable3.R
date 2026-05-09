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
#if the results don't make too much sense, consider using the interaction term version
#calculate variables importance metrics for the basic model (no interactions because first and last metrics can't be calculated)
get_lmg_with_ci <- function(model, savepath, terms = "lmg", nboot = 1000, level = 0.95) {
  
  # 1) Bootstrap
  set.seed(123)
  boot <- boot.relimp(model, b = nboot, type = terms )
  be <- booteval.relimp(boot, level = level, bty = "perc", norank=T, nodiff = T)
  saveRDS(be, file = savepath) 
  
  # 2) Bootstrap mean LMG estimate
  boot_lmg_point <- be@lmg * 100

  # 3) CIs
  lower <- be$lmg.lower * 100
  upper <- be$lmg.upper * 100
  
  # 5) Format nicely for input to tibble later
  sprintf("%.1f (%.1f-%.1f)", boot_lmg_point, lower, upper)
}
print('boostrap total path')
total_path_lm = lm(network_detectability ~  RR..mean.in.window + RRV..mean.in.window + RV..mean.in.window + HR..mean.in.window +PVI..mean.in.window + HRV..mean.in.window + SpO2..mean.in.window + Mean.FD..mean.in.window + strain + isoflurane_percent + sex +dex_conc + actual_ses_order + Time.after.isoflurane.change,
                  data = df_scaled)
lmg_total_path <- get_lmg_with_ci(total_path_lm, './total_path_be.rds', c("lmg", "first", "last"))

print('bootstrap total path interac')
total_path_lm_interac = lm(network_detectability ~  RR..mean.in.window + RRV..mean.in.window + RV..mean.in.window + HR..mean.in.window +PVI..mean.in.window + HRV..mean.in.window + SpO2..mean.in.window + Mean.FD..mean.in.window + strain*isoflurane_percent + sex*isoflurane_percent + strain*dex_conc + sex*dex_conc + actual_ses_order + Time.after.isoflurane.change,
                           data = df_scaled)

###################### save as png ##################################3
be_total_path <- readRDS(file = './total_path_be.rds')


regressors <-c('isoflurane', 'dexmedetomidine', 'session', 'RR', 'RRV', 'RV', 'HR', 'PVI', 'HRV', 'SpO2', 'mean FD', 'strain', 'sex', 'time after isoflurane change')
LMG <-lmg_total_path #this is modified during reviews, used to be relimp instead of relimp_interac
first<-as.character(round(be_total_path@first*100,1))
last<-as.character(round(be_total_path@last*100,1))
relimp_tibble <-tibble(regressors, LMG, first, last) |>
  arrange(factor(regressors, levels = c('strain', 'sex', 'session', 'isoflurane', 'dexmedetomidine', 'time after isoflurane change', "RR", "RRV", "RV", "HR", "HRV", "PVI", "SpO2", "mean FD"))) %>%
  add_row(regressors = 'total instantaneous', LMG = '15.7', first = '26', last = '9') 
write.csv(relimp_tibble, "./nd_r2_ci2.csv")

relimp_table <- gt(relimp_tibble)|>
  fmt_markdown(columns = everything()
  ) |>
  tab_header(
    title = md("**Table 3: Importance of regressors in predicting network detectability**"),
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
    source_note = md("The variance explained (R^2^) by a regressor is calculated as: the increase in model R^2^ when that regressor is added to the model. This value depends on the order in which that regressor is added to the model, thus we present 3 approachs. LMG is the average R^2^ across all orderings, confidence intervals estimated by bootstrapping are in brackets. 'First' is when that regressor is added first. 'Last' is when that regressor is added last. Values were computed with the relaimpo package (Groemping, 2007). Interaction terms and random effects are not shown as their 'first' and 'last' contribution cannot be calculated with the package.")
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
gtsave(relimp_table, "./nd_r2_table_ci2.png")



