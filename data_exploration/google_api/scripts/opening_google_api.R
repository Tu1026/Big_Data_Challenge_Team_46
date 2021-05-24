# Title     : Opens google API data files. THis is just a copy of open_all_df.R but with a modified opening function
# Objective :
# Created by: Alex
# Created on: 2021-05-24

#-----------Purpose
#Create an R list containing all dfs in a given folder.


#----------Changes b4 executing
path_target_folder <- "Big_Data_Challenge_Team_46/data_exploration/google_api/data"
# setwd("Big_Data_Challenge_Team_46")
getwd()

#---------Setup

if (!("tidyverse" %in% installed.packages())) {
  install.packages("tidyverse")
}
if (!("vroom" %in% installed.packages())) {
  install.packages("vroom")
}
if (!("stringr" %in% installed.packages())) {
  install.packages("stringr")
}


library(tidyverse)
library(vroom)
library(stringr)


#----------Function

names <- as.list(list.files(path_target_folder))
names(names) <- names
names

open_all_df <- function(name,path_target_folder) {
  return (name <- read.csv(file = paste0(path_target_folder,"/",name),row.names = NULL))
}

#----------Execution

split <- str_split(path_target_folder, "/", simplify = FALSE)
len <- length(split[[1]])
name <- str_replace_all(split[[1]][len], pattern = "[:punct:]", "_")


assign(paste0("l_",name),lapply(names, open_all_df, path_target_folder))
print(paste0("l_",name," was generated as a list of data frames"))

#----------rm
rm(len,names,path_target_folder,split,name)


