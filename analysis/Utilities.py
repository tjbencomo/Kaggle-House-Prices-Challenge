import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn_pandas import CategoricalImputer


def summarize_missing(df):
    return df.isnull().sum()[df.isnull().any()]

def percentage_missing(df):
    return df.isnull().sum().divide(df.shape[0]).multiply(100)[df.isnull().any()]


def drop_unecessary_columns(df, cols):
    # ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    return df.drop(cols, axis=1)

def drop_missing_rows(df):
    return df.dropna(axis=0,how='any', subset=['MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Electrical', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'])

def impute_missing(df, col_name):
    imputer = Imputer(missing_values= 'NaN', strategy = 'mean', axis=0)
    df[col_name] = imputer.fit_transform(df[col_name].values.reshape(-1,1))
    return df

def impute_categorical(df, col_name):
    imputer = CategoricalImputer()
    df[col_name] = imputer.fit_transform(df[col_name])
    return df

def impute_all_missing(df):
    missing_cols = list(summarize_missing(df).index.values)
    missing_cols
    col_info = {}
    for col in missing_cols:
        #print("{} : {}".format(col, df[col].dtype))
        col_info[col] = str(df[col].dtype)
    for col in col_info:
        if col_info[col] == "object":
            df = impute_categorical(df, col)        
        else:
            df = impute_missing(df, col)
    return df

def encodeAndStandardize(df):
    df = df.set_index('Id')
    dummized = pd.get_dummies(df.select_dtypes(include='object'))
    print("Dummized shape : {}".format(dummized.shape))
    scaler = preprocessing.StandardScaler()
    normalized = scaler.fit_transform(df.select_dtypes(exclude='object'))
    normalized = pd.DataFrame(normalized, index=df.index, columns = df.select_dtypes(exclude='object').columns)
    print("Normalized shape : {}".format(normalized.shape))
    combined = pd.concat([normalized, dummized], axis=1)
    print("Combined shape : {}".format(combined.shape))
    return combined