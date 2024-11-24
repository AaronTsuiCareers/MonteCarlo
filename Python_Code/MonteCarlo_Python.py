#!/usr/bin/env python
# coding: utf-8

# ## MSDS 460 Decision Analytics Assignment 4 Python

# In[72]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  ConfusionMatrixDisplay
overall_start_time = time.time()


# In[73]:


start_time = time.time()
file = "/Users/nikhil/Desktop/drug200.csv" 
drug = pd.read_csv(file)

end_time = time.time()

elapsed_time = end_time - start_time
drug.head()


# In[74]:


print(f"It took {elapsed_time} to pull the data")


# In[75]:


drug.isna().sum()


# In[76]:


start_time = time.time()
sns.pairplot(drug, hue = 'Drug')
end_time = time.time()
elapsed_time = end_time - start_time


# In[77]:


print(f"It took {elapsed_time} to create this graph")


# In[78]:


drug['Drug'].sum


# In[79]:


fig = sns.FacetGrid(drug, hue="Drug", aspect=3, palette="Set2")
fig.map(sns.kdeplot, "Na_to_K", shade=True)
fig.add_legend()


# In[80]:


start_time = time.time()

sex_counts = drug['Sex'].value_counts()
bp_counts = drug['BP'].value_counts()
drug_counts = drug['Drug'].value_counts()


def pie_with_percentages(data, labels=None, colors=None, title="Pie Chart with Percentages"):
    percentages = [f"{label}\n{count / sum(data) * 100:.1f}%" for label, count in zip(labels, data)]
    plt.pie(data, labels=percentages, colors=colors, autopct='%1.1f%%')
    plt.title(title)


sex_colors = ['pink', 'lightblue']
bp_colors = ["#FF6347", "#5B9BD5", "#D3D3D3"]
drug_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700']


fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot pie chart for Sex distribution
plt.sca(axs[0])
pie_with_percentages(sex_counts.values, labels=['Female', 'Male'], colors=sex_colors, title="Distribution of Gender")
plt.legend(['Female', 'Male'], loc="upper right")

plt.sca(axs[1])
pie_with_percentages(bp_counts.values, labels=bp_counts.index, colors=bp_colors, title="Distribution of Blood Pressure Levels")
plt.legend(bp_counts.index, loc="upper right")

plt.sca(axs[2])
pie_with_percentages(drug_counts.values, labels=drug_counts.index, colors=drug_colors, title="Distribution of Drugs")
plt.legend(drug_counts.index, loc="upper right")

plt.tight_layout()
plt.show()

end_time = time.time()

elapsed_time = end_time - start_time
print(f"It took {elapsed_time} seconds to run these graphs")


# In[81]:


start_time = time.time()

# Cross-tabulations
sex_drug_crosstab = pd.crosstab(drug['Sex'], drug['Drug'], margins=True, margins_name="Total")
print("Cross-tabulation between Sex and Drug:\n", sex_drug_crosstab)

bp_drug_crosstab = pd.crosstab(drug['BP'], drug['Drug'], margins=True, margins_name="Total")
print("\nCross-tabulation between BP and Drug:\n", bp_drug_crosstab)

cholesterol_drug_crosstab = pd.crosstab(drug['Cholesterol'], drug['Drug'], margins=True, margins_name="Total")
print("\nCross-tabulation between Cholesterol and Drug:\n", cholesterol_drug_crosstab)

# Heatmap for Sex vs Drug
plt.figure(figsize=(10, 6))
sns.heatmap(sex_drug_crosstab.iloc[:-1, :-1], annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title('Heatmap: Sex vs Drug')
plt.xlabel('Drug Class')
plt.ylabel('Sex')
plt.show()

# Heatmap for BP vs Drug
plt.figure(figsize=(10, 6))
sns.heatmap(bp_drug_crosstab.iloc[:-1, :-1], annot=True, cmap="Oranges", fmt="d", cbar=False)
plt.title('Heatmap: BP vs Drug')
plt.xlabel('Drug Class')
plt.ylabel('Blood Pressure')
plt.show()

# Heatmap for Cholesterol vs Drug
plt.figure(figsize=(10, 6))
sns.heatmap(cholesterol_drug_crosstab.iloc[:-1, :-1], annot=True, cmap="Greens", fmt="d", cbar=False)
plt.title('Heatmap: Cholesterol vs Drug')
plt.xlabel('Drug Class')
plt.ylabel('Cholesterol')
plt.show()

# End timing
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")


# ## Logistic Regression Monte Carlo Experiment 

# In[82]:


label_encoder = LabelEncoder()

drug['Sex'] = label_encoder.fit_transform(drug['Sex'])
drug['BP'] = label_encoder.fit_transform(drug['BP'])
drug['Cholesterol'] = label_encoder.fit_transform(drug['Cholesterol'])
drug['Drug'] = label_encoder.fit_transform(drug['Drug'])

train_times = []
accuracies = []
classification_reports = []
confusion_matrices = []


num_simulations = 1000

def simulate_drug_data():
    X = drug.drop(columns=['Drug'])  
    y = drug['Drug']
    return X, y

for i in range(num_simulations):
    X, y = simulate_drug_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)  
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    train_times.append(end_time - start_time)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    class_report = classification_report(y_test, y_pred)
    classification_reports.append(class_report)

    # Calculate and store the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

# Calculate averages
avg_train_time = np.mean(train_times)
avg_accuracy = np.mean(accuracies)

# Visualize the confusion matrix for the last iteration (or any specific one)
final_cm = confusion_matrices[-1]
disp = ConfusionMatrixDisplay(confusion_matrix=final_cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for the Logistic Regression")
plt.show()

# Print summary
print(f"Average Training Time: {avg_train_time:.4f} seconds")
print(f"Average Accuracy: {avg_accuracy:.4f}")


# ## Random Forest Monte Carlo Experiment 

# In[83]:


import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

num_simulations_rf = 100
train_times_rf = []
accuracies_rf = []

# Simulations
for i in range(num_simulations_rf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_model.fit(X_train, y_train)
    end_time = time.time()
    
    train_times_rf.append(end_time - start_time)
    
    y_pred_rf = rf_model.predict(X_test)
    
    accuracies_rf.append(accuracy_score(y_test, y_pred_rf))

confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix for Random Forest:")
print(confusion_matrix_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix Heatmap for Random Forest Model")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

avg_train_time_rf = np.mean(train_times_rf)
avg_accuracy_rf = np.mean(accuracies_rf)

print(f"Average training time for Random Forest: {avg_train_time_rf:.4f} seconds")
print(f"Average accuracy for Random Forest: {avg_accuracy_rf:.4f}")
overall_end_time = time.time()


# In[84]:


total_elapsed_time = overall_end_time - overall_start_time

print(f"\nTotal execution time for the entire script: {total_elapsed_time:.2f} seconds")

