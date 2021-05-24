# Title     : TODO
# Objective : TODO
# Created by: Alex
# Created on: 2021-05-24

library(infer)
library(skimr)
library(tidyverse)

null_hypothesis <- master_pro %>%
  specify(response = mean) %>%
  generate(reps = 100000, type = "bootstrap") %>%
  calculate(stat = 'mean')

null_vis <- ggplot(null_hypothesis) +
  geom_histogram(mapping = aes(stat), fill = "green", color = "white")+

null_stats <- skim(null_hypothesis)

print(null_stats$numeric.mean[2])


#
#hypothesis test
#

# hA <- master_pro%>%
#   specify(mean~Region) %>%
#   hypothesise(null = "independence")%>%
#   generate(reps = 1000, type = "permute")


#
# CI
#

null_ci <- null_hypothesis %>%
  get_confidence_interval(level = 0.95, type = 'percentile')

visualize(null_hypothesis)+
  shade_confidence_interval(endpoints = null_ci) +
  shade_p_value(78, "both")

