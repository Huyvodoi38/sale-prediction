import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Rossmann_df = pd.read_csv('C:/Users/DELL/Desktop/20242/Hoc may/Baitaplon/train.csv', low_memory=False)
store_df = pd.read_csv('C:/Users/DELL/Desktop/20242/Hoc may/Baitaplon/store.csv', low_memory=False)

data = pd.merge(Rossmann_df, store_df, on = 'Store', how ='left')

# 1. Phân tích tương quan
print("\n=== Phân tích tương quan ===")
# Tính toán ma trận tương quan cho các biến số
numeric_columns = data.select_dtypes(include=[np.number]).columns
correlation_matrix = data[numeric_columns].corr()

# Vẽ heatmap tương quan
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Ma trận tương quan giữa các biến số')
plt.tight_layout()
plt.show()

# 4. Kiểm tra tính nhất quán của dữ liệu
print("\n=== Kiểm tra tính nhất quán của dữ liệu ===")
# Kiểm tra giá trị null
print("\nSố lượng giá trị null trong mỗi cột:")
print(data.isnull().sum())

# Kiểm tra kiểu dữ liệu
print("\nKiểu dữ liệu của các cột:")
print(data.dtypes)

# Kiểm tra giá trị duy nhất trong các cột phân loại
categorical_columns = data.select_dtypes(include=['object']).columns
print("\nGiá trị duy nhất trong các cột phân loại:")
for col in categorical_columns:
    print(f"\n{col}:")
    print(data[col].value_counts())

# Kiểm tra tính nhất quán của dữ liệu thời gian
if 'Date' in data.columns:
    print("\nKiểm tra tính nhất quán của dữ liệu thời gian:")
    print("Khoảng thời gian từ:", data['Date'].min())
    print("đến:", data['Date'].max())
    print("Số ngày duy nhất:", data['Date'].nunique())

# Kiểm tra tính nhất quán của dữ liệu số
print("\nThống kê mô tả của các biến số:")
print(data[numeric_columns].describe())

sns.histplot(data['Sales'], bins=50)
plt.title('Phân bố doanh thu (Sales)')
plt.show()

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

data['Assortment'] = data['Assortment'].map({'a':0, 'c':1,'b':2})
data['Assortment'] = data['Assortment'].astype(int, copy=False)

data['StoreType'] = data['StoreType'].map({'a':1,'b':2,'c':3,'d':4})
data['StoreType'] = data['StoreType'].astype(int, copy=False)

data['PromoInterval'] = data['PromoInterval'].map({0: 0,'Jan,Apr,Jul,Oct': 1,'Feb,May,Aug,Nov': 2,'Mar,Jun,Sept,Dec': 3})
data['PromoInterval'] = data['PromoInterval'].astype(int, copy=False)

data['StateHoliday'] = data['StateHoliday'].replace("0",0)
data = pd.get_dummies(data, columns=["StateHoliday"],drop_first=True).astype(int)
