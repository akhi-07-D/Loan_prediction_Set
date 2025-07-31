# Loan_prediction_Set
This project uses Logistic Regression to predict whether a loan will be approved based on applicant information like income, education, employment status, and more.

📂 Dataset Information
The dataset contains details on loan applicants such as:
1.Categorical Features:
Gender, Married, Dependents, Education, Self_Employed, Property_Area, Loan_Status
2.Numerical Features:
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History

✅ Project Steps
1. Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

2. Load and Inspect Data
df = pd.read_csv("loan_data_set.csv")
df.head()
df.info()
df.isnull().sum()

3.Handle Missing Values
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

4.Convert Categorical Columns using One-Hot Encoding
df1 = pd.get_dummies(df, columns=["Gender", "Married", "Dependents", "Education","Self_Employed", "Property_Area", "Loan_Status"])

5.Drop redundant dummy columns to avoid multicollinearity
x = df1.iloc[:, 2:15]  # features
y = df1.iloc[:, -1]    # target

6. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

7. Train Logistic Regression Model
model = LogisticRegression()
model.fit(x_train, y_train)

8. Model Evaluation
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

9.Visualizations
sns.countplot(data=df, x='Loan_Status')
sns.barplot(data=df, x='Education', y='LoanAmount')
sns.jointplot(data=df, x='ApplicantIncome', y='LoanAmount', kind='scatter')
sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Loan_Status')
