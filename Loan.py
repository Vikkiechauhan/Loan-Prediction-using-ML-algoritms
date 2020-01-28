# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 23:29:07 2018

@author: Utsav
"""
import sys
import pandas as pd
import numpy as np
import matplotlib as plt
try:
    from Tkinter import *
except:
    from tkinter import *
window = Tk()
window.title("Loan Prediction")
window.geometry('350x200')
lbl = Label(window, text="Logistic Regression")
lbl.grid(column=10, row=0,padx=10, pady=10)


lbl2 = Label(window, text="Decision Tree")
lbl2.grid(column=10, row=10,padx=10)


lbl3 = Label(window, text="Random Forest")
lbl3.grid(column=10, row=20,padx=10)

lbl4 = Label(window, text="XG Boost")
lbl4.grid(column=10, row=30,padx=10)

df = pd.read_csv("trainfile.csv") 
test = pd.read_csv("testfile.csv")
df.head(10)
df.describe()
df['Gender'].value_counts()
df['Married'].value_counts()
df['Dependents'].value_counts() 
df['Education'].value_counts()
df['Self_Employed'].value_counts()
df['Credit_History'].value_counts()
df['Property_Area'].value_counts() 
df['Loan_Status'].value_counts()
df.apply(lambda x: sum(x.isnull()),axis=0)
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No',inplace=True)
table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
print(table)
def fage(x):
    return table.loc[x['Self_Employed'],x['Education']]

df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
df['Married'].value_counts()
df['Married'].fillna('Yes',inplace=True)
df['Dependents'].value_counts()
df['Dependents'].fillna('0',inplace=True)
df['Gender'].value_counts()
df['Gender'].fillna('Male',inplace=True)
df['Credit_History'].value_counts()
df['Credit_History'].fillna(1.0,inplace=True)
df['Loan_Amount_Term'].value_counts()
df['Loan_Amount_Term'].fillna(360.0,inplace=True)
test['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
from itertools import takewhile
test['LoanAmount_log'] = np.log(test['LoanAmount'])
test['TotalIncome'] = test['ApplicantIncome'] + test['CoapplicantIncome']
test['TotalIncome_log'] = np.log(test['TotalIncome'])
df['Dependents']=float(''.join(takewhile(str.isdigit, df['Dependents'])))
test['Dependents']=float(''.join(takewhile(str.isdigit, test['Dependents'])))
values_changed={"Gender": {"Male":1.0, "Female":0.0},
        "Education": {"Graduate":1.0, "Not Graduate":0.0},
         "Self_Employed": {"Yes":1.0, "No":0.0},
         "Married": {"Yes":1.0, "No":0.0},
        "Property_Area": {"Rural":1.0, "Semiurban":2.0, "Urban":3.0}}
df.replace(values_changed,inplace=True)
test.replace(values_changed,inplace=True)
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn import metrics
def classification_model(model, data, dtest, predictors, outcome):
    model.fit(data[predictors],data[outcome])
    predictions = model.predict(data[predictors])
    dtest[outcome] = model.predict(dtest[predictors])
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print ("Accuracy : " + str(accuracy))
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        train_predictors = (data[predictors].iloc[train,:])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    print ("Cross-Validation Score :" + str((np.mean(error))))
    dtest.to_csv("C:/Self learning/Projects/Loan Prediction/submission.csv")
    #Logistic regression-->significant variables
def LR():  
    outcome_var = 'Loan_Status'
    model = LogisticRegression()
    predictor_var = ['Gender','Credit_History','Education','Married','Self_Employed','Property_Area']
    classification_model(model, df,test,predictor_var,outcome_var)
    

    
    #Decision Tree--> with high important features based on feature importance graph obtained
def DT():
    from sklearn.tree import DecisionTreeClassifier
    outcome_var = 'Loan_Status'
    model = DecisionTreeClassifier()
    predictor_var = ['Credit_History', 'LoanAmount_log','TotalIncome_log']

    classification_model(model, df,test,predictor_var,outcome_var)
    featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
    
    
    
    

     
    #Random Forest
def RF():
    from sklearn.ensemble import RandomForestClassifier
    outcome_var = 'Loan_Status'
    model = RandomForestClassifier(n_estimators=100)
    predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
    classification_model(model, df,test,predictor_var,outcome_var)
    featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
    outcome_var = 'Loan_Status'
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, max_depth=7, max_features=1)
    predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
    classification_model(model, df,test,predictor_var,outcome_var)

def XGB():
    from xgboost import XGBClassifier
    outcome_var='Loan_Status'
    model= XGBClassifier(n_estimators=100)
    predictor_var=['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
    classification_model(model, df,test,predictor_var,outcome_var)
btn1 = Button(window, text="Check Accuracy", command=LR)
btn1.grid(column=20, row=0, padx=100, pady=10)

btn2 = Button(window, text="Check Accuracy", command=DT)
btn2.grid(column=20, row=10, padx=5, pady=10)

btn3 = Button(window, text="Check Accuracy", command=RF)
btn3.grid(column=20, row=20, padx=5, pady=10)

btn4 = Button(window, text="Check Accuracy", command=XGB)
btn4.grid(column=20, row=30, padx=5, pady=10)

window.mainloop()
