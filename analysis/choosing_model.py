import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import pickle
import sys

def loadData():
    data = pd.read_csv('/Users/Tomas/Desktop/Kaggle-House-Prices-Challenge/data/final_train.csv', index_col=0)
    prices = data['SalePrice']
    data = data.drop(['SalePrice'], axis=1)

    poly = PolynomialFeatures(2)
    data = poly.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(data, prices, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test



def linear_regression(X_train, X_test, y_train, y_test):
    linear_regressor = Pipeline([('pca', PCA(.98)), 
                        ('regr', linear_model.LinearRegression())])

    estimator = GridSearchCV(linear_regressor, cv=5, n_jobs=-1, param_grid={}, scoring='neg_mean_squared_error', verbose=2)

    estimator.fit(X_train, y_train)
    print("Linear Regression Baseline Cross Validation Score {}".format(estimator.best_score_))
    predictions = estimator.predict(X_test)
    print("Linear Regression Baseline Test Score {}".format(mean_squared_error(y_test, predictions)))
    
    return estimator

def elasticnet_regression(X_train, X_test, y_train, y_test):
    elasticnet_regression = Pipeline([('pca', PCA(.98)), 
                        ('regr', linear_model.ElasticNet())])
    estimator = GridSearchCV(elasticnet_regression, cv=5, n_jobs=2, param_grid={}, scoring='neg_mean_squared_error', verbose=1)

    estimator.fit(X_train, y_train)
    print("ElasticNet Regression Baseline Cross Validation Score {}".format(estimator.best_score_))
    predictions = estimator.predict(X_test)
    print("ElasticNet Regression Baseline Test Score {}".format(mean_squared_error(y_test, predictions)))

    return estimator

def bayesian_regression(X_train, X_test, y_train, y_test):
    bayesian_regression = Pipeline([('pca', PCA(.98)), 
                        ('regr', linear_model.BayesianRidge())])
    estimator = GridSearchCV(bayesian_regression, cv=5, n_jobs=-1, param_grid={}, scoring='neg_mean_squared_error', verbose=1)

    estimator.fit(X_train, y_train)
    print("BayesianRidge Regression Baseline Cross Validation Score {}".format(estimator.best_score_))
    predictions = estimator.predict(X_test)
    print("BayesianRidge Regression Baseline Test Score {}".format(mean_squared_error(y_test, predictions)))

    return estimator


def svr_regression(X_train, X_test, y_train, y_test):
    svr_regression = Pipeline([('pca', PCA(.98)), 
                        ('regr', svm.SVR(kernel='linear'))])
    estimator = GridSearchCV(svr_regression, cv=5, n_jobs=-1, param_grid={}, scoring='neg_mean_squared_error', verbose=1)

    estimator.fit(X_train, y_train)
    print("Support Vector Machine Regression Baseline Cross Validation Score {}".format(estimator.best_score_))
    predictions = estimator.predict(X_test)
    print("Support Vector Machine Regression Baseline Test Score {}".format(mean_squared_error(y_test, predictions)))

    return estimator
    
def rf_regression(X_train, X_test, y_train, y_test):
    rf_regression = Pipeline([('pca', PCA(.98)), 
                        ('regr', RandomForestRegressor())])
    estimator = GridSearchCV(rf_regression, cv=5, n_jobs=-1, param_grid={}, scoring='neg_mean_squared_error', verbose=1)

    estimator.fit(X_train, y_train)
    print("Random Forest Regression Baseline Cross Validation Score {}".format(estimator.best_score_))
    predictions = estimator.predict(X_test)
    print("Random Forest Regression Baseline Test Score {}".format(mean_squared_error(y_test, predictions)))

    return estimator

def tune_bayesian(X_train, X_test, y_train, y_test):
    bayesian_regression = Pipeline([('pca', PCA(.98)), 
                        ('regr', linear_model.BayesianRidge())])

    param_grid = {'regr__alpha_1' : [1e-6, 1e-4, 1e-3, 1e-1],
                    'regr__alpha_2' : [1e-6, 1e-4, 1e-3, 1e-1],
                    'regr__lambda_1' : [1e-6, 1e-4, 1e-3, 1e-1],
                    'regr__lambda_2' : [1e-6, 1e-4, 1e-3, 1e-1]}

   
    estimator = GridSearchCV(bayesian_regression, cv=3, n_jobs=-1, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=1)
    # print(estimator.get_params().keys())
    # sys.exit(0)
    estimator.fit(X_train, y_train)
    print("BayesianRidge Regression Hypertuned Cross Validation Score {}".format(estimator.best_score_))
    print("Best parameters : {}".format(estimator.best_params_))
    predictions = estimator.predict(X_test)
    print("BayesianRidge Regression Hypertuned Test Score {}".format(mean_squared_error(y_test, predictions)))

    return estimator

def main():
    X_train, X_test, y_train, y_test = loadData()
    # linreg = linear_regression(X_train, X_test, y_train, y_test)
    # elasticnet = elasticnet_regression(X_train, X_test, y_train, y_test)
    # bayesian = bayesian_regression(X_train, X_test, y_train, y_test)
    # svr = svr_regression(X_train, X_test, y_train, y_test)
    # rf = rf_regression(X_train, X_test, y_train, y_test)
    # knn regressor, randomforest

    tuned_regresssor = tune_bayesian(X_train, X_test, y_train, y_test)
    
    with open('/Users/Tomas/Desktop/Kaggle-House-Prices-Challenge/models/bayesian_tuned_regressor.pkl', 'wb') as fid:
        pickle.dump(tuned_regresssor, fid)

if __name__ == '__main__':
    main()
