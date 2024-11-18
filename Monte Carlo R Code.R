start_time <- Sys.time()
# Load necessary liinstabraries
library(tidyverse)
library(caret)
library(randomForest)
library(GGally)
library(ggplot2)
library(reshape2)
library(RColorBrewer)


# Set the file path and read the CSV
start_time_benchmark <- Sys.time()
file <- "drug200.csv"
drug <- read.csv(file)
drugend_time <- Sys.time()
elapsed_time <- drugend_time - start_time_benchmark
print(paste("It took", elapsed_time, "seconds to pull the data"))

# Check for NA values
print(sum(is.na(drug)))

# Pairplot equivalent in R (using GGally package)
start_time <- Sys.time()
ggpairs(drug, aes(color = Drug))
end_time <- Sys.time()
elapsed_time <- end_time - start_time
print(paste("It took", elapsed_time, "seconds to create this graph"))

# Drug counts
print(table(drug$Drug))

# Density plot for Na_to_K by Drug
ggplot(drug, aes(x = Na_to_K, fill = Drug)) + 
  geom_density(alpha = 0.5) + 
  theme_minimal() + 
  labs(title = "Distribution of Na_to_K by Drug")

# Pie charts for categorical distributions
pie_chart <- function(data, labels, colors, title) {
  pie(data, labels = paste(labels, round(100 * data / sum(data), 1), "%"), col = colors, main = title)
}

par(mfrow = c(3, 3))
sex_counts <- table(drug$Sex)
bp_counts <- table(drug$BP)
drug_counts <- table(drug$Drug)

pie_chart(sex_counts, names(sex_counts), c("pink", "lightblue"), "Distribution of Gender")
pie_chart(bp_counts, names(bp_counts), c("#FF6347", "#5B9BD5", "#D3D3D3"), "Distribution of Blood Pressure Levels")
pie_chart(drug_counts, names(drug_counts), c('#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700'), "Distribution of Drugs")

# Cross-tabulations
sex_drug_crosstab <- table(drug$Sex, drug$Drug)
bp_drug_crosstab <- table(drug$BP, drug$Drug)
cholesterol_drug_crosstab <- table(drug$Cholesterol, drug$Drug)

# Convert crosstabs to data frames
sex_drug_df <- as.data.frame(sex_drug_crosstab)
bp_drug_df <- as.data.frame(bp_drug_crosstab)
cholesterol_drug_df <- as.data.frame(cholesterol_drug_crosstab)

# Create heatmap for Sex vs Drug
ggplot(sex_drug_df, aes(x = Var2, y = Var1, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black") +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Heatmap: Sex vs Drug", x = "Drug Class", y = "Sex") +
  theme_minimal()

# Create heatmap for BP vs Drug
ggplot(bp_drug_df, aes(x = Var2, y = Var1, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black") +
  scale_fill_gradient(low = "white", high = "darkorange") +
  labs(title = "Heatmap: BP vs Drug", x = "Drug Class", y = "Blood Pressure") +
  theme_minimal()

# Create heatmap for Cholesterol vs Drug
ggplot(cholesterol_drug_df, aes(x = Var2, y = Var1, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black") +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(title = "Heatmap: Cholesterol vs Drug", x = "Drug Class", y = "Cholesterol") +
  theme_minimal()

# Define features and target
drug$Sex <- as.numeric(as.factor(drug$Sex))
drug$BP <- as.numeric(as.factor(drug$BP))
drug$Cholesterol <- as.numeric(as.factor(drug$Cholesterol))
drug$Drug <- ifelse(drug$Drug == "DrugY", 1, 0)

X <- drug[, c('Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K')]
y <- drug$Drug

# Monte Carlo Experiment for Logistic Regression
num_simulations <- 1000
train_times_log <- numeric(num_simulations)
accuracies_log <- numeric(num_simulations)

for (i in 1:num_simulations) {
  trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
  X_train <- X[trainIndex, ]
  X_test <- X[-trainIndex, ]
  y_train <- y[trainIndex]
  y_test <- y[-trainIndex]
  
  start_time <- Sys.time()
  log_model <- glm(y_train ~ ., data = as.data.frame(X_train), family = binomial)
  end_time <- Sys.time()
  
  train_times_log[i] <- as.numeric(end_time - start_time)
  y_pred_log <- predict(log_model, newdata = as.data.frame(X_test), type = "response")
  y_pred_log <- ifelse(y_pred_log > 0.5, 1, 0)
  accuracies_log[i] <- mean(y_pred_log == y_test)
}

# Confusion Matrix for Logistic Regression
confusion_matrix_log <- table(y_test, y_pred_log)
print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix_log)

# Visualizing the confusion matrix for Logistic Regression
confusion_df_log <- as.data.frame(confusion_matrix_log)
colnames(confusion_df_log) <- c("Actual", "Predicted", "Count")

ggplot(confusion_df_log, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "darkblue") +
  geom_text(aes(label = Count), color = "white", size = 5) +
  labs(title = "Confusion Matrix Heatmap for Logistic Regression Model",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

avg_train_time_log <- mean(train_times_log)
avg_accuracy_log <- mean(accuracies_log)

print(paste("Average training time for Logistic Regression:", avg_train_time_log))
print(paste("Average accuracy for Logistic Regression:", avg_accuracy_log))

# Random Forest Monte Carlo Experiment
num_simulations_rf <- 100
train_times_rf <- numeric(num_simulations_rf)
accuracies_rf <- numeric(num_simulations_rf)

for (i in 1:num_simulations_rf) {
  trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
  X_train <- X[trainIndex, ]
  X_test <- X[-trainIndex, ]
  y_train <- y[trainIndex]
  y_test <- y[-trainIndex]
  
  start_time <- Sys.time()
  rf_model <- randomForest(x = X_train, y = as.factor(y_train), ntree = 100)
  end_time <- Sys.time()
  
  train_times_rf[i] <- as.numeric(end_time - start_time)
  y_pred_rf <- predict(rf_model, newdata = X_test)
  accuracies_rf[i] <- mean(y_pred_rf == as.factor(y_test))
}

# After predicting with the random forest model
y_pred_rf <- predict(rf_model, newdata = X_test)

# Confusion Matrix for Random Forest
confusion_matrix_rf <- table(y_test, y_pred_rf)
print("Confusion Matrix for Random Forest:")
print(confusion_matrix_rf)

# Visualizing the confusion matrix for Random Forest
confusion_df_rf <- as.data.frame(confusion_matrix_rf)
colnames(confusion_df_rf) <- c("Actual", "Predicted", "Count")

ggplot(confusion_df_rf, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "darkblue") +
  geom_text(aes(label = Count), color = "white", size = 5) +
  labs(title = "Confusion Matrix Heatmap for Random Forest Model",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

avg_train_time_rf <- mean(train_times_rf)
avg_accuracy_rf <- mean(accuracies_rf)

print(paste("Average training time for Random Forest:", avg_train_time_rf))
print(paste("Average accuracy for Random Forest:", avg_accuracy_rf))

end_time_benchmark <- Sys.time()
elapsed_time_benchmark <- end_time_benchmark - start_time_benchmark
print(paste("It took", elapsed_time_benchmark, "seconds to run the benchmark"))
