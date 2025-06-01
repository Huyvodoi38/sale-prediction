import numpy as np
import pandas as pd

data = pd.read_csv('C:/Users/DELL/Desktop/20242/Hoc may/Baitaplon/merged_output.csv', low_memory=False)

# Replacing Null values with 0 in CompetitionDistance
data['CompetitionDistance'] = data['CompetitionDistance'].fillna(data['CompetitionDistance'].median())

# Replacing Null values with 0 in CompetitionOpenSinceMonth
data['CompetitionOpenSinceMonth'] = data['CompetitionOpenSinceMonth'].fillna(0)

# Replacing Null values with 0 in CompetitionOpenSinceYear
data['CompetitionOpenSinceYear'] = data['CompetitionOpenSinceYear'].fillna(0)

# Replacing Null values with 0 in Promo2SinceWeek
data['Promo2SinceWeek'] = data['Promo2SinceWeek'].fillna(0)

## Replacing Null values with 0 in Promo2SinceYear
data['Promo2SinceYear'] = data['Promo2SinceYear'].fillna(0)

## Replacing Null values with 0 in PromoInterval
data['PromoInterval'] =data['PromoInterval'].fillna(0)

# first 'Date' object column
# Extracting year, month and day from "Date" using pd.to_datetime
# and Droping column 'Date
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].apply(lambda x: x.year)
data['Month'] = data['Date'].apply(lambda x: x.month)
data['Day'] = data['Date'].apply(lambda x: x.day)

data.drop('Date',axis=1,inplace=True)

object_columns = {'StateHoliday','StoreType','Assortment','PromoInterval'}

# Changing Assortment, StoreType and PromoInterval Datatypes ti int
data['Assortment'] = data['Assortment'].map({'a':0, 'c':1,'b':2})
data['Assortment'] = data['Assortment'].astype(int, copy=False)

data['StoreType'] = data['StoreType'].map({'a':1,'b':2,'c':3,'d':4})
data['StoreType'] = data['StoreType'].astype(int, copy=False)

data['PromoInterval'] = data['PromoInterval'].map({0: 0,'Jan,Apr,Jul,Oct': 1,'Feb,May,Aug,Nov': 2,'Mar,Jun,Sept,Dec': 3})
data['PromoInterval'] = data['PromoInterval'].astype(int, copy=False)

data['StateHoliday'] = data['StateHoliday'].replace("0",0)

# Encoding Stateholiday
data = pd.get_dummies(data, columns=["StateHoliday"],drop_first=True).astype(int)

data.to_csv('output.csv', index=False)  # index=False để không ghi chỉ mục