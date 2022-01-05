#!/usr/bin/env python
# coding: utf-8

# 7 Steps of Machine Learning
# 
#         1: Gathering Data.
#         2: Preparing that Data.
#         3: Choosing a Model.
#         4: Training.
#         5: Evaluation.
#         6: Hyperparameter Tuning.
#         7: Prediction.

# In[88]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
#rsme == mse**(.5)

import warnings
warnings.filterwarnings('ignore')


# Gathering Data

# In[89]:


data = pd.read_csv("/Users/harikrishnareddy/Desktop/NYCdataset.csv")


# In[90]:


print("Shape of the DataSet : \n",data.shape,
      "\n Dimensions in the dataset: \n",data.ndim)


# In[91]:


print("Types in the dataSet : \n",data.dtypes)


# Preparing that Data.

# In[92]:


data.isnull().sum()


# There are no Missing values in the data 

# In[93]:


data['vendor_id'].value_counts()


# In[94]:


data['store_and_fwd_flag'].value_counts() #unbalenced


# Label Encoding
# 
#     refers to converting the labels into a numeric form so as to convert them into the machine-readable form

# In[95]:


data['pickup_datetime'][0]


# In[96]:


data['pickup_datetime'][1]


# In[97]:


count = len(data.index)


# In[99]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['encodingFlag'] = le.fit_transform(data['store_and_fwd_flag'])


# In[ ]:


data['encodingFlag'].value_counts()


# N -> 0
# 
#         It has converted too..
# y -> 1

# In[ ]:


print("Data in Seconds \n",data['trip_duration'].describe())
#convert seconds to minutes
print("\nData in Minutes \n",data['trip_duration'].describe()/60)


# In[ ]:


#convert minutes to hours
data['trip_duration_hours'] = data['trip_duration']/(60*60)


# we have date and time so we can sepearate them to use

# In[ ]:


data['pickup_datetime'] = pd.to_datetime(data.pickup_datetime)
data['dropoff_datetime'] = pd.to_datetime(data.dropoff_datetime)
data['phourOfDay'] = data['pickup_datetime'].dt.hour
data['pdayinWeek'] = data['dropoff_datetime'].dt.weekday


# In[ ]:


data.head(2)


# In[ ]:


#log normalize
data['td_log'] = np.log(data['trip_duration'].values + 1)


# In[ ]:


data['trip_duration_hours'].plot.box() #oulier in trip duration hours


# In[ ]:


data[data['trip_duration_hours']<=24]#removing outlier
data['trip_duration_hours'].max() ,data['trip_duration_hours'].min()


# In[ ]:


#to build model we need remove the columns which are not useful
data.head(1)


# In[ ]:


newdata = data.drop(columns =['id','pickup_datetime', 'dropoff_datetime', 'pickup_longitude','pickup_latitude',
                       'dropoff_longitude','dropoff_latitude','store_and_fwd_flag','trip_duration'])


# In[ ]:


newdata


# In[ ]:


X = newdata.drop('td_log',axis=1)
y = newdata['td_log']


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)
X = pd.DataFrame(x_scaled, columns=X.columns)


# Choosing a Model.

# 1. Choose the most suitable evaluation metric and state why you chose it.

# In general we have two types of metrics
# 
#     1. Classification
#     2. Regression    
#     
# Clearly we can say ours is a regression type
# In regression we have diffrent types 
#        
#      MAE, MSE, RMSE and RMSLE
# If we go by defination RMSE will be more efficient I guess(commonly used)
# 
#     

# In[ ]:


#using train_test_split splitting data into trainingSet and testSet
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)


# Buliding a Regression Model

# K Nearest Neighbours
# 
#     1.It is a supervised learning algorithm.
# 
#     2.kNN is very simple to implement and is most widely used as a first step in many machine learning setup
# 
#     3.The number of nearest neighbours to a new unknown variable(test vaaiable) that has to be predicted or classified is denoted by the symbol 'K'

# In[118]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(xtrain, ytrain)
tmp_pred = reg.predict(xtest)
temp_error = MSE(tmp_pred, ytest)


# In[ ]:


def elbow(k):
  test = []

  for i in k:
    reg = KNeighborsRegressor(n_neighbors=i)
    reg.fit(xtrain, ytrain)
    tmp_pred = reg.predict(xtest)
    temp_error = MSE(tmp_pred, ytest)
    test.append(temp_error)

  return test


# In[ ]:


k = [1,2,3,4,5,6,7,8,9,10]


# In[ ]:


test = elbow(k)
# plotting the curve

plt.plot(k, test)
plt.xlabel('K Neighbors')
plt.ylabel('RMSE')
plt.title('Elbow curve for test')


# In[ ]:


knn = KNeighborsRegressor(n_neighbors=4)
knn.fit(xtrain, ytrain)
y_pred = knnr.predict(xtest)
knn_test_rmse = MSE(y_test, y_pred)

print("RMSE of knn model: ", knn_test_rmse)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

lm_test_rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE of linear regressor model: ", lm_test_rmse)

y_pred = lr.predict(X_train)

lm_rmse = sqrt(mean_squared_error(y_train, y_pred))

print("RMSE of linear regressor model: ", lm_rmse)


# In[ ]:




