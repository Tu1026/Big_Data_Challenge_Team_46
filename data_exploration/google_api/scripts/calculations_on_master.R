# Title     : TODO
# Objective : TODO
# Created by: Alex
# Created on: 2021-05-24
install.packages("qwraps2")
install.packages("infer")

library(qwraps2)
library(infer)
library(skimr)

glimpse(master)


rownames(master) <- as.list(master[,1])

master_pro <- master %>%
  rowwise(Region) %>%
  mutate(mean = mean(c_across(), na.rm = TRUE))
glimpse(master_pro)

summary_stats <- skimr::skim(master_pro$mean)
summary_stats$numeric.mean
