# When we are given 10 hours to build a model,we will work 6 hours to build the dataset ~dataScientist

# UnderStand the dataset by
# 
#     check for null/missing values and replace
#     check for outliers
#     
# and perform your analysis further

# Importing all the nessecary packages for work



import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Analysing given dataset

#Reading the dataSet using panda
data = pd.read_csv("/Users/harikrishnareddy/Desktop/nyc_taxi_trip_duration Dataset.csv")




#lets see what we got in here like shape,types and look for any null values or duplicates in dataSet
print(data.shape)
print(f'In the given dataSet the No.of rows are {data.shape[0]} & columns are {data.shape[1]}')


print(data.dtypes)


# id - a unique identifier for each trip
# 
# vendor_id - a code indicating the provider associated with the trip record
# 
# pickup_datetime - date and time when the meter was engaged
# 
# dropoff_datetime - date and time when the meter was disengaged
# 
# passenger_count - the number of passengers in the vehicle (driver entered value)
# 
# pickup_longitude - the longitude where the meter was engaged
# 
# pickup_latitude - the latitude where the meter was engaged
# 
# dropoff_longitude - the longitude where the meter was disengaged
# 
# dropoff_latitude - the latitude where the meter was disengaged
# 
# store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server (Y=store and forward; N=not a store and forward trip)
# 
# trip_duration - (target) duration of the trip in seconds


print("The number of null values in each column\n",data.isnull().sum())


# As we can see from the dtypes we should change the few given types
# 
#     pickup_datetime,dropoff_datetime are  objects converting to datetime
# 
# this will help to predict the data more accurate


data['pickup_datetime'] = pd.to_datetime(data.pickup_datetime)
data['dropoff_datetime'] = pd.to_datetime(data.dropoff_datetime)
data[['pickup_datetime','dropoff_datetime']].dtypes



data['store_and_fwd_flag'].value_counts()


# we shall plot the graph of all the continous variables


print(data['trip_duration'].describe(),"\n")
#If you remember the dataSet trip_duratoin is seconds so we will convert them seonds or hours by dividing it 3600(hours)
print('\n',data['trip_duration'].describe()/3600)


# what 538 hours,this is our outlier in the given data for 'trip_duration'


sns.distplot(data['trip_duration'])
plt.show()


# If this was in bell curve we could have analyzed but it not in that way,we can use 
# 
#      LOG Transform
#      When our original continuous data do not follow the bell curve(consits of outliers), we can log transform this data to make it as 'normal', so that our statistical analysis becomes more valid


plt.figure(figsize=(20, 5))



#using log
data['tripLog'] = np.log(data['trip_duration'])
plt.subplot(111)
sns.distplot(data['tripLog'], kde = False, bins = 150)
plt.show()

#this may result in negative to avoid we will add 1 
data['tripLog'] = np.log(data['trip_duration'].values+1)
#plt.subplot(123)
sns.distplot(data['tripLog'])
plt.show()



plt.figure(figsize=(20, 10))
plt.subplot(222)
sns.countplot(data['vendor_id'])
plt.xlabel('vendor_id')
plt.ylabel('Frequency')
plt.show()
print("vendor 2 has suplied more cabs")
plt.figure(figsize=(10, 5))
sns.countplot(data['store_and_fwd_flag'])
plt.xlabel('store_and_fwd_flag')
plt.ylabel('Frequency')


# we shall find the corr between numerical columns
data.corr()
plt.figure(figsize=(12, 6))
df1 = data.drop(['id','tripLog','store_and_fwd_flag','trip_duration'],
        axis=1)
ax = sns.heatmap(data.corr(), xticklabels=df1.columns, yticklabels=df1.columns, 
                 linewidths=.2)


# As we have datatime type in our dataset we can do some analysis on them !

#we should get data and time seperate from datetime.

#geting the day in the week
data['dw'] = data['pickup_datetime'].dt.weekday

#getting the hour in the day
data['hd'] = data['pickup_datetime'].dt.hour


# Datetime features
plt.figure(figsize=(20, 5))

# Passenger Count
plt.subplot(121)
sns.countplot(data['dw'])
plt.xlabel('Week Day')
plt.ylabel('Total Number of pickups')

# vendor_id
plt.subplot(122)
sns.countplot(data['hd'])
plt.xlabel('Hour')
plt.ylabel('Total number of pickups')


# Points :
# 
#     0-Sunday.....6-Saturday
# Sunday and Saturday are having low trips when we compare with weekdays and also high on Thursday.And trips are high in evening hour in the day. 18th and 19th hours 
