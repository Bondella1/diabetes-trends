import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

#For K-fold
from sklearn.model_selection import cross_val_score,KFold
from sklearn.svm import SVC 

#Check for co-linearity

def main():
    #X_train.csv`/`y_train.csv

    df = pd.read_csv("data/processed/X_train.csv")
    df = df[["inactivity_prevalence","obesity_prevalence","smoking_prevalence"]]

    df.head()

    corrmatrix = df.corr(method= 'pearson')
    print(corrmatrix)
    print('\n-----------------------------------------------------------------------------')
    
        
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    X_train = X_train[['inactivity_prevalence','obesity_prevalence',"smoking_prevalence"]]
    X_test = X_test[['inactivity_prevalence','obesity_prevalence',"smoking_prevalence"]]


    model1 = LinearRegression()
    model1.fit(X_train,y_train)

    y_pred = model1.predict(X_test)


    print(f'\nCoefficients:\n\tInactivity: {model1.coef_[0][0]}\n\tObesity: {model1.coef_[0][1]}\n\tSmoking: {model1.coef_[0][2]}\n'
          f'Intercept: {model1.intercept_[0]}\n'
          f'R-Squared: {r2_score(y_test,y_pred)}\n'
          f'Mean Squared Error: {mean_squared_error(y_test,y_pred)}\n'
          f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}\n')
    
    print('\n-----------------------------------------------------------------------------')
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cross_val_results = cross_val_score(model1, X_train, y_train, cv=kf)

    print("Cross-Validation Results:")
    for i, result in enumerate(cross_val_results,1):
        print(f'\tFold {i}: {result*100:.2f}%')
    print(f'Mean Accuracy: {cross_val_results.mean()*100:.2f}%')

main()
#colinearity seems fine, model *may* be done? But check more on conflating the two inact and obes, also understand how the 3d reg plane works. 