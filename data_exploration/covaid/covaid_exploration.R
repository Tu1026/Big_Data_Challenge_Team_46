# Title     :
# Objective :
# Created by: Alex
# Created on: 2021-05-22
setwd("Big_Data_Challenge_Team_46")
library(tidyverse)

#run open_all_df.R before

glimpse(l_11_01_2020[[1]])
names(l_11_01_2020)

glimpse(l_11_01_2020[["NewsRealCOVID-19_tweets_replies.csv"]])

News_real_replies <- l_11_01_2020[["NewsRealCOVID-19_tweets_replies.csv"]]
print (News_real_replies$tweet_id, digits = 20)