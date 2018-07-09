import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder
import Utilities
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import numpy as np

print("-------------------TRAINING-------------------")

# load data
train = pd.read_csv('/Users/Tomas/Desktop/Kaggle-House-Prices-Challenge/data/train.csv')
print(train.shape)
init_train_cols = set(train.columns)

# remove columns with too much data missing
train = Utilities.drop_unecessary_columns(train)
print(train.shape)

# impute rows with missing values
train = Utilities.impute_all_missing(train)
print(train.shape)

# encode and standardize
prices = train["SalePrice"]
prices = np.log(prices)
train = Utilities.encodeAndStandardize(train)
print("Train shape : {}".format(train.shape))
train['SalePrice'] = prices
print(train.shape)

print("-------------------TESTING-------------------")

# load data
test = pd.read_csv('/Users/Tomas/Desktop/Kaggle-House-Prices-Challenge/data/test.csv')
print(test.shape)
init_test_cols = set(test.columns)

# print(Utilities.summarize_missing(test))

# remove columns with too much data missing
test = Utilities.drop_unecessary_columns(test)
print(test.shape)

# impute rows with missing data 
test = Utilities.impute_all_missing(test)
print(test.shape)

test = Utilities.encodeAndStandardize(test)
print(test.shape)

train_cols = set(train.columns)
test_cols = set(test.columns)
print(train_cols - test_cols)
print()
print(init_train_cols - init_test_cols)
