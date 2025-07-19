# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# Read the CSV file
data <- read_csv("imputation_results.csv")

# Display the structure of the data
str(data)
head(data)

# Convert MissingRate to factor for better grouping
data$MissingRate <- as.factor(data$MissingRate)
data$Method <- as.factor(data$Method)

# Create color palette for methods
method_colors <- c("MICE" = "#E74C3C", "FAMD" = "#3498DB", 
                   "missForest" = "#2ECC71", "MIDAS" = "#F39C12")

# 1. RMSE Comparison by Missing Rate and Method
p1 <- ggplot(data, aes(x = MissingRate, y = RMSE, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  geom_text(aes(label = round(RMSE, 2)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.3, size = 3) +
  scale_fill_manual(values = method_colors) +
  labs(title = "RMSE Comparison by Missing Rate and Method",
       x = "Missing Rate",
       y = "RMSE",
       fill = "Method") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

# 2. PFC Comparison by Missing Rate and Method
p2 <- ggplot(data, aes(x = MissingRate, y = PFC, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  geom_text(aes(label = round(PFC, 3)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.3, size = 3) +
  scale_fill_manual(values = method_colors) +
  labs(title = "PFC Comparison by Missing Rate and Method",
       x = "Missing Rate",
       y = "PFC",
       fill = "Method") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

# 3. TimeSec Comparison by Missing Rate and Method (Log scale for better visualization)
p3 <- ggplot(data, aes(x = MissingRate, y = TimeSec, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  geom_text(aes(label = round(TimeSec, 1)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.3, size = 3) +
  scale_fill_manual(values = method_colors) +
  scale_y_log10() +
  labs(title = "Execution Time Comparison by Missing Rate and Method (Log Scale)",
       x = "Missing Rate",
       y = "Time (seconds) - Log Scale",
       fill = "Method") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

# Display individual plots
print(p1)
print(p2)
print(p3)

# 4. Alternative combined view using par() for multiple plots
# Set up plotting area for 3 plots in one window
par(mfrow = c(3, 1), mar = c(4, 4, 2, 1))

# 5. Alternative: Faceted comparison of all metrics
# Reshape data for faceted plot
data_long <- data %>%
  select(MissingRate, Method, RMSE, PFC, TimeSec) %>%
  pivot_longer(cols = c(RMSE, PFC, TimeSec), 
               names_to = "Metric", 
               values_to = "Value")

# Create faceted plot
p4 <- ggplot(data_long, aes(x = MissingRate, y = Value, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  facet_wrap(~Metric, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = method_colors) +
  labs(title = "Performance Metrics Comparison by Missing Rate and Method",
       x = "Missing Rate",
       y = "Value",
       fill = "Method") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        strip.text = element_text(size = 12, face = "bold"))

print(p4)

# 6. Line plot to show trends across missing rates
p5 <- ggplot(data, aes(x = as.numeric(as.character(MissingRate)), color = Method, group = Method)) +
  geom_line(aes(y = RMSE), size = 1.2) +
  geom_point(aes(y = RMSE), size = 3) +
  scale_color_manual(values = method_colors) +
  labs(title = "RMSE Trends Across Missing Rates",
       x = "Missing Rate",
       y = "RMSE",
       color = "Method") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

print(p5)

# 7. Summary statistics table
summary_stats <- data %>%
  group_by(Method) %>%
  summarise(
    Avg_RMSE = round(mean(RMSE), 3),
    Avg_PFC = round(mean(PFC), 3),
    Avg_TimeSec = round(mean(TimeSec), 1),
    .groups = 'drop'
  )

print("Summary Statistics by Method:")
print(summary_stats)

# 8. Best performing method by metric and missing rate
best_performance <- data %>%
  group_by(MissingRate) %>%
  summarise(
    Best_RMSE = Method[which.min(RMSE)],
    Best_PFC = Method[which.min(PFC)],
    Fastest = Method[which.min(TimeSec)],
    .groups = 'drop'
  )

print("Best Performing Methods by Missing Rate:")
print(best_performance)

# Save plots as PDFs
ggsave("rmse_comparison.pdf", p1, width = 10, height = 6)
ggsave("pfc_comparison.pdf", p2, width = 10, height = 6)
ggsave("time_comparison.pdf", p3, width = 10, height = 6)
ggsave("combined_metrics.pdf", p4, width = 12, height = 10)
ggsave("rmse_trends.pdf", p5, width = 10, height = 6)

# Alternative: Save all plots in a single multi-page PDF
pdf("all_plots.pdf", width = 12, height = 8)
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
dev.off()

print("All plots saved as individual PDFs and combined in 'all_plots.pdf'")