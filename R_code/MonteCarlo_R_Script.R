start_time <- Sys.time()
# Load necessary libraries 
library(tidyverse)
library(caret)
library(randomForest)
library(GGally)
library(ggplot2)
library(reshape2)
library(RColorBrewer)
library(nnet)


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
create_pie_chart <- function(data, labels, colors, title) {
  pie(data, 
      labels = paste(labels, round(100 * data / sum(data), 1), "%"), 
      col = colors, 
      main = title, 
      border = "white", 
      radius = 0.8)
}

# Set up the plotting area
par(mfrow = c(1, 3))

# Create pie charts
create_pie_chart(sex_counts, names(sex_counts), c("pink", "lightblue"), "Distribution of Gender")
create_pie_chart(bp_counts, names(bp_counts), c("#FF6347", "#5B9BD5", "#D3D3D3"), "Distribution of Blood Pressure Levels")
create_pie_chart(drug_counts, names(drug_counts), c('#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700'), "Distribution of Drugs")
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

# Encode categorical variables as factors
drug$Sex <- as.factor(drug$Sex)
drug$BP <- as.factor(drug$BP)
drug$Cholesterol <- as.factor(drug$Cholesterol)
drug$Drug <- as.factor(drug$Drug)

# Prepare the data for modeling
X <- drug[, c('Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K')]
y <- drug$Drug

# Monte Carlo Experiment for Logistic Regression
num_simulations <- 1000
train_times_log <- numeric(num_simulations)
accuracies_log <- numeric(num_simulations)

set.seed(42)

for (i in 1:num_simulations) {
  # Split data into train and test (80/20 split)
  trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
  X_train <- X[trainIndex, ]
  X_test <- X[-trainIndex, ]
  y_train <- y[trainIndex]
  y_test <- y[-trainIndex]
  
  start_time <- Sys.time()
  log_model <- glm(y_train ~ ., data = cbind(X_train, y_train), family = binomial)
  end_time <- Sys.time()
  
  train_times_log[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Predicting probabilities for Logistic Regression
  y_pred <- predict(log_model, newdata = X_test, type = "response")
  y_pred_class <- ifelse(y_pred > 0.5, levels(y)[2], levels(y)[1])  # Convert probabilities to class predictions
  
  accuracy <- mean(y_pred_class == y_test)
  accuracies_log[i] <- accuracy
}

# Confusion Matrix for Logistic Regression
confusion_matrix_log <- table(y_test, y_pred_class)
print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix_log)

# Visualizing the confusion matrix for Logistic Regression
confusion_df_log <- as.data.frame(as.table(confusion_matrix_log))
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
  y_test <- 