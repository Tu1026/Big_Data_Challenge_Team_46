# Title     : vis
# Objective : TODO
# Created by: Alex
# Created on: 2021-05-24

glimpse(master_pro)

master_pro <- master_pro %>%
  arrange(desc(mean))
master_pro$Region <- factor(master_pro$Region, levels = master_pro$Region)
master_pro$Region

my_plot <- ggplot(master_pro) +
  geom_col(mapping = aes(y = master_pro$mean,x = master_pro$Region), color = "black", fill = "light green") +
  geom_hline(mapping = aes(yintercept= summary_stats$numeric.mean), color = "red")+
  labs(title = "Covid 19 Hoax Related Searches")+
  xlab ("State")+
  ylab("Mean Search Count (Normalized to 100)") +
  theme(axis.text.x = element_text(angle = 90))
my_plot