import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
def lr(train):
    fl = train
    x = train.iloc[:, :-1]  
    y = train.iloc[:, -1] 
    X_train , X_test , y_train, y_test = train_test_split(x,y,test_size=0.2)
    model =linear_model.LinearRegression()
    model.fit(X_train,y_train)
    y_pred_train = model.predict(X_test)



    return X_test,y_test,y_pred_train,model