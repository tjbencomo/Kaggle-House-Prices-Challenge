import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import Models, Utilities
from sklearn.model_selection import train_test_split



PATH = "/Users/Tomas/Desktop/Kaggle-House-Prices-Challenge/data/"  #where you put the files
df_train = pd.read_csv(f'{PATH}train.csv', index_col='Id')
df_test = pd.read_csv(f'{PATH}test.csv', index_col='Id')
target = df_train['SalePrice']  #target variable
df_train = df_train.drop('SalePrice', axis=1)
df_train['training_set'] = True
df_test['training_set'] = False
df_full = pd.concat([df_train, df_test])
print(Utilities.summarize_missing(df_full))
df_full = df_full.interpolate()
print("INTERPOLATED")
print(Utilities.summarize_missing(df_full))

df_full = pd.get_dummies(df_full)
print("\n\n\n")
print(Utilities.summarize_missing(df_full))
df_train = df_full[df_full['training_set']==True]
df_train = df_train.drop('training_set', axis=1)
df_test = df_full[df_full['training_set']==False]
df_test = df_test.drop('training_set', axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_train, target, test_size=.3, random_state=42)

rf = Models.rfRegression(X_train, X_test, y_train, y_test)

Models.generate_predictions(rf, 'RandomForest', 'cv_with_solution', df_test)

# rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
# rf.fit(df_train, target)
# preds = rf.predict(df_test)
# my_submission = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds})
# my_submission.to_csv(f'{PATH}submission.csv', index=False)