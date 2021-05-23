# Title     : open_all_df
# Objective : Open all data frames in a folder
# Created by: Alex
# Created on: 2021-05-22

#-----------Purpose
#Create an R list containing all dfs in a given folder.


#----------Changes b4 executing
path_target_folder <- "alex_src/CoAID-master/07-01-2020"

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
  return (name <- vroom::vroom(paste0(path_target_folder,"/",name)))
}

#----------Execution

split <- str_split(path_target_folder, "/", simplify = FALSE)
len <- length(split[[1]])
name <- str_replace_all(split[[1]][len], pattern = "[:punct:]", "_")


assign(paste0("l_",name),lapply(names, open_all_df, path_target_folder))
print(paste0("l_",name," was generated as a list of data frames"))

#----------rm
rm(len,names,path_target_folder,split,name)