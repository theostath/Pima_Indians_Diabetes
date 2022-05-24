## Step 1: Reading the Data

# Imports

import numpy as np
import pandas as pd # for data analysis
import matplotlib.pyplot as plt
import seaborn as sns   # for statistical graphics
from sklearn.model_selection import train_test_split


# Reading the data-frame

df = pd.read_csv('diabetes.csv')

## Step 2: Exploratory Data Analysis

# Take a look at the different variables of the dataset, which are:
#Pregnancies - Number of times pregnant
#Glucose - Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#Blood Pressure - Diastolic blood pressure
#SkinThickness - Triceps skin fold thickness
#Insulin - 2-Hour serum insulin
#BMI - Body Mass Index
#DiabetesPedigreeFunction - Diabetes pedigree function
#Age - Age
#Outcome - Whether or not the person is diabetic

# Print the first 5 rows
print(df.head())

# Outcome is our TARGET, and all other variables are the predictors.
# Predict the outcome with a ML model using the remaining variables.

# Take a look at some descriptive statistics.
print(df.describe())

# View the relationship betweeen all the variables in the dataset.
# pairplot:
sns.pairplot(df, hue='Outcome')
plt.show()

# Pair plot allows you to take a look at the relationship bewtween all the variables in the dataset at once.

# You can then go on to examine these relationships, and remove highly correlated attributes that might cause any multicollinearity in your model.

## Step 3: Data Pre-processing

# Check for missing values in the DataFrame
print(df.isnull().values.any())

# This should return False.

# Standardize the variables to get all of them around the same scale.
# Z-Score standardization: This will take all our variables and transform them to follow a normal distribution having a mean of 0 and a standard deviation of 1.

X = df.drop('Outcome', axis=1) # define the predictor variables
y = df['Outcome'] # define the target variable

# Standardization (x-μ)/σ
X = (X-X.mean())/X.std()

# Now inspect the head of the DataFrame again.
print(X.head())

## Step 4: Build the Machine Learning Model

# First, we need to split the DataFrame into a train and test set.
# We will train the model on one set, and the evaluate its performance on unseen data.

# One of the most popular ways to perform model validation is K-fold cross validation.

# Here, we will cover the simplest validation approach: we will divide the dataset into two, and hold out one set of data for testing.

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25) # 25% of dataset size is chosen for the test size.

## Logistic Regression Classifier

# It uses this linear function: y = a + bX to separate values of each class.
# It modifies the linear function a little such that the values will not go below 0 or above 1.

# As a result, logistic regression classifier outputs PROBABILITIES. By default, the model prediction is the class with the highest probability.

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)

lr_preds = lr.predict(X_test)


## Decision Tree Classifier

# It is a tree-based model. It will take all the variables in the dataset and split on them.

# At each split, it either assigns the data point to a class or splits on a different variable. 
# This process continues until there are no more features to split on, or a stopping criteria is reached.

# The decision tree classifier decides what to split on by selecting the feature that minimizes the loss at each split. 
# The Gini index and entropy are two common loss functions used in decision tree classifiers.

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

dt_preds = dt.predict(X_test)


## Random Forest Classifier

# This model uses multiple decision trees to come up with a prediction.
# In practice, random forests often outperform decision trees and linear classifiers.

# It uses a technique called bagging, which stand for bootstrap aggregation.
# The training dataset is randomly sampled many times, and a decision tree is fit onto each data sample.

# Only a subset of features are considered for division at each node, ensuring a fair representation of each variable in the model.

# In classification problems like this one, the output is the majority class prediction of all the decision trees.

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

rf_preds = rf.predict(X_test)


## XGBoost Classifier

# One of the most popular supervised learning technique, and it used often by data scientists in Kaggle competitions.
# In practice, they tend to perform a lot better than decision trees.

# This classifier uses a technique called boosting to improve the performance of decision trees.

# The way gradient boosting algorithms work is pretty intuitive.
# A base learner is first implemented that makes an initial prediction. 
# The residuals of the initial model is calculated, and a second decision tree is added, that predicts the residuals of the initial model.

# A chain of decision trees are added to form a sequential model that minimizes the overall residual.

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)

xgb_preds = xgb.predict(X_test)


## Step 5: Evaluation

# Simplest metric to evaluate model performance: accuracy

model = np.array(['Logistic Regression','Decision Tree','Random Forest','Gradient Boosting'])

from sklearn.metrics import accuracy_score

scores = np.array([accuracy_score(lr_preds,y_test),accuracy_score(dt_preds,y_test),accuracy_score(rf_preds,y_test),accuracy_score(xgb_preds,y_test)])

df = {'model': model, 'scores': scores}
sns.barplot(x='model',y='scores',data=df)
plt.show()

print('Accuracy of Logistic Regression:', 100*accuracy_score(lr_preds,y_test),'%')
print('Accuracy of Decision Tree:', 100*accuracy_score(dt_preds,y_test),'%')
print('Accuracy of Random Forest:', 100*accuracy_score(rf_preds,y_test),'%')
print('Accuracy of Gradient Boosting:', 100*accuracy_score(xgb_preds,y_test),'%')