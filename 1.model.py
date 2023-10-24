# Import necessary libraries
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, classification_report
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# Read the dataset from a CSV file
dataset = pd.read_csv("Bank Customer Churn Prediction.csv")
print("------------------------------------------------------------------------------------------")
print(dataset.head())

print(dataset.shape)

print(dataset.describe())
print("------------------------------------------------------------------------------------------")

# Drop the 'customer_id' column, as it's not needed for modeling
dataset = dataset.drop("customer_id", axis=1)

print(dataset.isnull().sum())
print("------------------------------------------------------------------------------------------")

print(dataset['churn'].value_counts())
print("------------------------------------------------------------------------------------------")

# Use One-Hot Encoding to convert categorical features 'country' and 'gender' into numeric
encoder = OneHotEncoder(sparse_output=False)
country_encoded = encoder.fit_transform(dataset[['country', 'gender']])


# Concatenate the one-hot encoded columns with the original dataset
newDataset = pd.concat([dataset.drop(columns=['country', 'gender']), pd.DataFrame(
    country_encoded, columns=encoder.get_feature_names_out(['country', 'gender']))], axis=1)


# Scale selected numeric features to a common range
scaler = MinMaxScaler()


features_to_be_scaled = ["credit_score", "age", "tenure",
                         "balance", "products_number", "estimated_salary"]

newDataset[features_to_be_scaled] = scaler.fit_transform(
    newDataset[features_to_be_scaled])

# Loop through each feature to check the number of unique values
for column in dataset:
    uniques_vals = np.unique(dataset[column])
    nr_vals = len(uniques_vals)
    if nr_vals < 36:
        print(
            f"The number of values for feature {column} : {nr_vals} -- {uniques_vals}")
    else:
        print(f"The number of values for feature {column} : {nr_vals}")
print("------------------------------------------------------------------------------------------")

# Limiting the data to include numeric columns
numeric_dataset = dataset[["credit_score", "age",
                           "balance", "estimated_salary", "churn"]]

sns.countplot(x="churn", data=dataset, palette='hls')
plt.title("Frequency of clients")
plt.show()

# calculate correlation between columns
corr_matrix = numeric_dataset.corr()
print(corr_matrix)

# Visualize the data using seaborn pairplots
sns.pairplot(numeric_dataset, hue='churn', diag_kws={'bw_method': 0.2})
plt.show()

# Define a list of features to create countplots for
features = ["country", "gender", "tenure", "products_number",
            "credit_card", "active_member"]

# Create countplots for each feature
for f in features:
    plt.figure()
    ax = sns.countplot(x=f, data=dataset, hue="churn", palette='Set1')
    plt.show()

# Split the dataset into features (X) and target (y)
X = newDataset.drop("churn", axis=1)
y = newDataset["churn"].values

# Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes
sm = SMOTE(random_state=42)
nr = NearMiss()

# Split the dataset into train and test proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train, y_train = sm.fit_resample(X_train, y_train.ravel())

# Checking if the classes are equal
count_0 = 0
count_1 = 0
for i in y_train:
  if i == 0:
    count_0 += 1
  else:
    count_1 += 1
print(f"0: {count_0}")
print(f"1: {count_1}")

# Checking the shape of each predictor
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("------------------------------------------------------------------------------------------")

# Define parameter space for GridSearchCV
parameter_space = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive']
}

# Create a GridSearchCV object for the MLPClassifier
optimized_neural_network = GridSearchCV(MLPClassifier(
    random_state=42), parameter_space, n_jobs=1, cv=3)

print("------------------------------------------------------------------------------------------")
# Print information about the Logistic Regression model
print("Logistic Regression Model: ")
print("------------------------------------------------------------------------------------------")
lr = LogisticRegression(random_state=42)

lr.fit(X_train, y_train)


y_predictions = lr.predict(X_test)
print(f"Accuracy  = {round(accuracy_score(y_test, y_predictions)*100, 2)}%")
print(f"Precision = {round(precision_score(y_test, y_predictions)*100,2)}%")
print(f"Recall = {round(recall_score(y_test, y_predictions)*100,2)}%")
print(f"F1-Score = {round(f1_score(y_test, y_predictions)*100,2)}%")

print(classification_report(y_true=y_test, y_pred=y_predictions))
print("------------------------------------------------------------------------------------------")

# Print information about the Naive Bayes model
print("------------------------------------------------------------------------------------------")
print("Naive Bayes Model: ")
print("------------------------------------------------------------------------------------------")
nb = GaussianNB()

nb.fit(X_train, y_train)


y_predictions = nb.predict(X_test)
print(f"Accuracy  = {round(accuracy_score(y_test, y_predictions)*100, 2)}%")
print(f"Precision = {round(precision_score(y_test, y_predictions)*100,2)}%")
print(f"Recall = {round(recall_score(y_test, y_predictions)*100,2)}%")
print(f"F1-Score = {round(f1_score(y_test, y_predictions)*100,2)}%")

print(classification_report(y_true=y_test, y_pred=y_predictions))
print("------------------------------------------------------------------------------------------")

# Print information about the Decision Tree model
print("------------------------------------------------------------------------------------------")
print("Decision Tree Model: ")
print("------------------------------------------------------------------------------------------")
nb = DecisionTreeClassifier(random_state=42)

nb.fit(X_train, y_train)


# Important Variables
print("Important Features: ")
for i, column in enumerate(newDataset.drop("churn", axis=1)):
    #print(f"Importance of feature {column}:, {nb.feature_importances_[i]}")
    fi = pd.DataFrame({'variable': [column], 'Feature Importance Score': [
                      nb.feature_importances_[i]]})

    try:
        final_fi = pd.concat([final_fi, fi], ignore_index=True)
    except:
        final_fi = fi
        
# ordering the data
final_fi = final_fi.sort_values("Feature Importance Score", ascending=False).reset_index()
print(final_fi)

print()
print("Performance measures: ")

y_predictions = nb.predict(X_test)

print(f"Accuracy  = {round(accuracy_score(y_test, y_predictions)*100, 2)}%")
print(f"Precision = {round(precision_score(y_test, y_predictions)*100,2)}%")
print(f"Recall = {round(recall_score(y_test, y_predictions)*100,2)}%")
print(f"F1-Score = {round(f1_score(y_test, y_predictions)*100,2)}%")

print(classification_report(y_true=y_test, y_pred=y_predictions))
print("------------------------------------------------------------------------------------------")

# Print information about the Random Forest model
print("------------------------------------------------------------------------------------------")
print("Random Forest Model: ")
print("------------------------------------------------------------------------------------------")
nb = RandomForestClassifier(random_state=42)

nb.fit(X_train, y_train)


y_predictions = nb.predict(X_test)
print(f"Accuracy  = {round(accuracy_score(y_test, y_predictions)*100, 2)}%")
print(f"Precision = {round(precision_score(y_test, y_predictions)*100,2)}%")
print(f"Recall = {round(recall_score(y_test, y_predictions)*100,2)}%")
print(f"F1-Score = {round(f1_score(y_test, y_predictions)*100,2)}%")

print(classification_report(y_true=y_test, y_pred=y_predictions))
print("------------------------------------------------------------------------------------------")

# Print information about the Support Vector Machine model
print("------------------------------------------------------------------------------------------")
print("Support Vector Machine Model: ")
print("------------------------------------------------------------------------------------------")
nb = SVC(random_state=42)

nb.fit(X_train, y_train)


y_predictions = nb.predict(X_test)
print(f"Accuracy  = {round(accuracy_score(y_test, y_predictions)*100, 2)}%")
print(f"Precision = {round(precision_score(y_test, y_predictions)*100,2)}%")
print(f"Recall = {round(recall_score(y_test, y_predictions)*100,2)}%")
print(f"F1-Score = {round(f1_score(y_test, y_predictions)*100,2)}%")

print(classification_report(y_true=y_test, y_pred=y_predictions))
print("------------------------------------------------------------------------------------------")

print("------------------------------------------------------------------------------------------")
# Print information about the Linear Discriminant Analysis model
print("Linear Discriminant Analysis Model: ")
print("------------------------------------------------------------------------------------------")
nb = LinearDiscriminantAnalysis()

nb.fit(X_train, y_train)


y_predictions = nb.predict(X_test)
print(f"Accuracy  = {round(accuracy_score(y_test, y_predictions)*100, 2)}%")
print(f"Precision = {round(precision_score(y_test, y_predictions)*100,2)}%")
print(f"Recall = {round(recall_score(y_test, y_predictions)*100,2)}%")
print(f"F1-Score = {round(f1_score(y_test, y_predictions)*100,2)}%")

print(classification_report(y_true=y_test, y_pred=y_predictions))
print("------------------------------------------------------------------------------------------")

# Print information about the Quadratic Discriminant Analysis model
print("------------------------------------------------------------------------------------------")
print("Quadratic Discriminant Analysis Model: ")
print("------------------------------------------------------------------------------------------")
nb = QuadraticDiscriminantAnalysis()

nb.fit(X_train, y_train)


y_predictions = nb.predict(X_test)
print(f"Accuracy  = {round(accuracy_score(y_test, y_predictions)*100, 2)}%")
print(f"Precision = {round(precision_score(y_test, y_predictions)*100,2)}%")
print(f"Recall = {round(recall_score(y_test, y_predictions)*100,2)}%")
print(f"F1-Score = {round(f1_score(y_test, y_predictions)*100,2)}%")

print(classification_report(y_true=y_test, y_pred=y_predictions))
print("------------------------------------------------------------------------------------------")

# Print information about the Gradient Boosting model
print("------------------------------------------------------------------------------------------")
print("Gradient Boosting Model: ")
print("------------------------------------------------------------------------------------------")
nb = GradientBoostingClassifier(random_state=42)

nb.fit(X_train, y_train)


y_predictions = nb.predict(X_test)
print(f"Accuracy  = {round(accuracy_score(y_test, y_predictions)*100, 2)}%")
print(f"Precision = {round(precision_score(y_test, y_predictions)*100,2)}%")
print(f"Recall = {round(recall_score(y_test, y_predictions)*100,2)}%")
print(f"F1-Score = {round(f1_score(y_test, y_predictions)*100,2)}%")

print(classification_report(y_true=y_test, y_pred=y_predictions))
print("------------------------------------------------------------------------------------------")

# Print information about the K-nearest neighbor model
print("------------------------------------------------------------------------------------------")
print("K-Nearest Neighbor Model: ")
print("------------------------------------------------------------------------------------------")
nb = KNeighborsClassifier()

nb.fit(X_train, y_train)


y_predictions = nb.predict(X_test)
print(f"Accuracy  = {round(accuracy_score(y_test, y_predictions)*100, 2)}%")
print(f"Precision = {round(precision_score(y_test, y_predictions)*100,2)}%")
print(f"Recall = {round(recall_score(y_test, y_predictions)*100,2)}%")
print(f"F1-Score = {round(f1_score(y_test, y_predictions)*100,2)}%")

print(classification_report(y_true=y_test, y_pred=y_predictions))
print("------------------------------------------------------------------------------------------")

# Print information about the Neural Network model
print("------------------------------------------------------------------------------------------")
print("Multilayer Perceptron Model: ")
print("------------------------------------------------------------------------------------------")
nb = MLPClassifier(random_state=42)

nb.fit(X_train, y_train)


y_predictions = nb.predict(X_test)
print(f"Accuracy  = {round(accuracy_score(y_test, y_predictions)*100, 2)}%")
print(f"Precision = {round(precision_score(y_test, y_predictions)*100,2)}%")
print(f"Recall = {round(recall_score(y_test, y_predictions)*100,2)}%")
print(f"F1-Score = {round(f1_score(y_test, y_predictions)*100,2)}%")

print(classification_report(y_true=y_test, y_pred=y_predictions))
print("------------------------------------------------------------------------------------------")

# Assuming you have true labels and predicted probabilities
precision, recall, _ = precision_recall_curve(y_test, y_predictions)
cf = confusion_matrix(y_test, y_predictions)


# Plot the confusion matrix
sns.heatmap(cf, annot=True, fmt='g', xticklabels=[
            'NOT CHURNED', 'CHURNED'], yticklabels=['NOT CHURNED', 'CHURNED'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

# Plot the Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()
