# Pima_Indians_Diabetes
Kaggle problem: Predict the onset of diabetes based on diagnostic measures.

# Create andactivate the virtual environment
First, create an environment with conda where we will pre-install python 3.9:

```
conda create --name PimaIndianDiabetes python=3.9
```

Then activate the environment and change directory:

```
conda activate PimaIndianDiabetes

cd anaconda3/envs/PimaIndianDiabetes
```

# Instal the prerequisites

Use pip install to install the packages needed to run the code. These are the following:
pandas, numpy, matplotlib, seaborn, scikit-learn and xgboost.

# Run the code

Then you can download main.py and run it in the terminal:

```
python main.py
```

# Dataset info

The dataset has many different variables:
Pregnancies - Number of times pregnant

Glucose - Plasma glucose concentration a 2 hours in an oral glucose tolerance test

Blood Pressure - Diastolic blood pressure

SkinThickness - Triceps skin fold thickness

Insulin - 2-Hour serum insulin

BMI - Body Mass Index

DiabetesPedigreeFunction - Diabetes pedigree function

Age - Age

Outcome - Whether or not the person is diabetic

The variable 'Outcome' is the target and all the other variables are the predictors. 
Therefore, we use the predictors in four different ML models to find out the target value.

# ML models

First, we standardize the variables to follow a normal distribution with a mean of 0 and standard deviation of 1.
We do that by substracting the mean value and dividing with the standard deviation of each variable.

Then, we split the data into two sets, training and testing.

Then we apply these four ML models:
Logistic Regression Classifier

Decision Tree Classifier

Random Forest Classifier

XGBoost Classifier

# Evaluation

Finally, we plot the accuracy of each model.
