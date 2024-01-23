# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:07:11 2023

@author: EMMANUEL
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import precision_recall_curve

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import pprint
from collections import Counter
import string
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("SMSSpam.csv", index_col=False)


# #remove rows with any values that are not finite
# data = df[np.isfinite(df).all(1)]




print("length of values")
print(len(data))


print("count values")
print(data.label.value_counts())

data['length'] = data['text'].apply(len)
data.head()





# #Tokenization

documents =  data['text']

lower_case_documents = []
lower_case_documents = [d.lower() for d in documents]
print(lower_case_documents)


sans_punctuation_documents = []


for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(str.maketrans("","", string.punctuation)))
    
print(sans_punctuation_documents)

print("  ")

preprocessed_documents = [[w for w in d.split()] for d in sans_punctuation_documents]
print(preprocessed_documents)
print("  ")



# Count frequencies

frequency_list = []


frequency_list = [Counter(d) for d in preprocessed_documents]
pprint.pprint(frequency_list)

print("  ")



# count_vector = CountVectorizer()




# count_vector.fit(documents)

# print("get feature names")
# print(count_vector.get_feature_names())

# print("  ")
# doc_array = count_vector.transform(documents).toarray()
# print(doc_array)
# print("  ")





# frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())
# print(frequency_matrix)
# print("  ")





# Summary statistics
print(data.describe())

print(" ")
# Display basic information about the dataset
print(data.info())
print(" ")
# Check for missing values
print(data.isnull().sum())

print(" ")
# Correlation matrix
correlation_matrix = data.corr()
print(correlation_matrix)


# Assuming your dataset has two columns: 'label' and 'text'
X = data['text']
y = data['label']

# Convert labels to binary (spam: 1, ham: 0)
y = y.map({'spam': 1, 'ham': 0})


data.loc[:,'label'] = data.label.map({'spam': 1, 'ham': 0})
print(data.shape)
data.head()


print(data.head())
print(" ")




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)











svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_tfidf, y_train)




# Make predictions on the test set
y_pred = svm_model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", accuracy)

# Generate a classification report
class_report = classification_report(y_test, y_pred)
print("SVM Classification Report:\n", class_report)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("SVM Confusion Matrix:\n", conf_matrix)






# Create a K-Nearest Neighbors classifier (you can adjust the 'n_neighbors' parameter)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Fit the model on the training data
knn_model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = knn_model.predict(X_test_tfidf)

# Calculate accuracy and print a classification report
accuracy1 = accuracy_score(y_test, y_pred)
print(" K-Nearest Neighbors Accuracy:", accuracy1)
print(" K-Nearest Neighbors Classification Report:\n", classification_report(y_test, y_pred))

# Print a confusion matrix
conf_matrix1 = confusion_matrix(y_test, y_pred)
print(" K-Nearest Neighbors Confusion Matrix:\n", conf_matrix)



sns.heatmap(conf_matrix1, annot=True, fmt="d", cmap="flag")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("KNN Confusion Matrix")
plt.show()




# Create a confusion matrix heatmap
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()



# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, svm_model.decision_function(X_test_tfidf))

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Calculate AUC (Area Under the ROC Curve)
auc = roc_auc_score(y_test, svm_model.decision_function(X_test_tfidf))
print("AUC:", auc)



# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, svm_model.decision_function(X_test_tfidf))

# Plot precision-recall curve
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()





#To display a histogram of material input features, you can use Matplotlib's hist function. Add the following code after the SVM model fitting:
    
    
    
    
    # Plot a histogram of input features
plt.figure()
plt.hist(X_train_tfidf.toarray().flatten(), bins=50, color='dodgerblue', alpha=0.7)
plt.xlabel('TF-IDF Values')
plt.ylabel('Frequency')
plt.title('Histogram of Material Input Features')
plt.show()


# Plot a pie chart of the outcome distribution
plt.figure()
labels = ['Ham', 'Spam']
sizes = [len(y_train[y_train == 0]), len(y_train[y_train == 1])]
colors = ['lightcoral', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Pie Chart of Outcome Distribution in Training Data')
plt.show()


# Lists to store train and validation error
train_errors = []
validation_errors = []

# Vary the regularization parameter C and store errors for different values
C_values = [0.001, 0.01, 0.1, 1, 10]
for C in C_values:
    svm_model = SVC(kernel='linear', C=C)
    svm_model.fit(X_train_tfidf, y_train)
    y_train_pred = svm_model.predict(X_train_tfidf)
    y_val_pred = svm_model.predict(X_test_tfidf)
    train_errors.append(1 - accuracy_score(y_train, y_train_pred))
    validation_errors.append(1 - accuracy_score(y_test, y_val_pred))

# Plot train and validation error
plt.figure()
plt.plot(C_values, train_errors, marker='o', label='Train Error', color='orange')
plt.plot(C_values, validation_errors, marker='o', label='Validation Error', color='blue')
plt.xscale('log')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Error')
plt.title('Train and Validation Error vs. Regularization Parameter')
plt.legend()
plt.grid(True)
plt.show()





