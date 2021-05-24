# Title     : merge all google_api data by state
# Objective :
# Created by: Alex
# Created on: 2021-05-24

library(tidyverse)
library(stringr)

getwd()
data_dir <-  "Big_Data_Challenge_Team_46/data_exploration/google_api/data"

glimpse(l_data$covid_19_hoax.csv)

df <- l_data$covid_19_hoax.csv
as.character(df[1,2])
df_un <- unlist(df$Category..All.categories)

as.numeric(as.character(df$Category..All.categories))

#
# Function for cleaning google api data
#
clean_goog_api <- function(df) {
  colnames(df) <- c("Region",as.character(df[1,2]))
  df <- df[-c(1),]
  df[,2] <- as.numeric(as.character(df[,2]))
  return (df)
}

#
# Executing clean_goog_api
#

l_data_clean <- lapply(l_data,clean_goog_api)

#
# Merge all databases
#

merge_to_master <- function(l_data_clean) {
  states <- c("New Mexico", "Indiana","West Virginia","New Hampshire" ,
"Iowa","Tennessee", "Minnesota","Idaho",
"Maine","Washington","Missouri","Oklahoma"  ,
"Colorado","Nevada","Michigan","Mississippi"    ,
"North Carolina","California","Georgia","South Carolina" ,
"Arizona","Virginia", "Nebraska","Maryland"    ,
"Wisconsin","Texas","Kentucky","Massachusetts",
"Ohio","Oregon","Florida", "Utah"  ,
"Illinois", "Kansas", "New York","Pennsylvania" ,
"Alabama","Arkansas", "New Jersey","Connecticut" ,
"Louisiana" ,"Wyoming","Delaware","Montana",
"Alaska","Vermont","Rhode Island","Hawaii",
"North Dakota","South Dakota","District of Columbia")
  master <- data.frame(Region = states )
  colnames(master) <- c("Region")
  for (i in l_data_clean){
    master <- left_join(master ,i, by = "Region")
  }
  return (master)
}

master <- merge_to_master(l_data_clean)
glimpse(master)
length(master$Region)



