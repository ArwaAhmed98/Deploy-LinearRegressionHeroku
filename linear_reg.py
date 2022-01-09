
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Any results you write to the current directory are saved as output.

#Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
np.set_printoptions(threshold=np.inf, linewidth=np.nan)


#Importing DataSet 
dataset = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/swe/kc_house_data.csv")
space=dataset['sqft_living'] #space 
price=dataset['price'] # vs price 

x = np.array(space).reshape(-1, 1)
y = np.array(price)
# array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
#Splitting the data into Train and Test
#from sklearn.cross_validation import train_test_split 
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0) # divide your DATA into train and test


#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Predicting the prices
pred = regressor.predict(xtest)

