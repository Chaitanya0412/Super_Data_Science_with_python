#Data Preprocessing

#import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset

dataset =pd.read_csv('Data.csv')
X =dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#handling missing values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
(X[:, 1:3]) = imputer.fit_transform(X[:, 1:3])

# handling Categorical Data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0]=labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#Splittinf Data in Training set and test set

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test =train_test_split (X,Y,test_size = 0.2,random_state=0 )


#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)
