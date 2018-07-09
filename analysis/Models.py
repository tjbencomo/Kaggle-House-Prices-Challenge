from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import Utilities
from sklearn.ensemble import RandomForestRegressor


def rmse(predictions, truths):
    return math.sqrt(mean_squared_error(truths, predictions))

def linearRegression(X_train, X_test, y_train, y_test):
    LoS = Pipeline([('pca', PCA(.95)), 
                     ('regr', linear_model.LinearRegression())])
    LoS.fit(X_train, y_train)
    predictions = LoS.predict(X_test)
    print("LoS Regression:")
    print("Test RMSE : {}".format(rmse(predictions, y_test)))
    print()
    
    return LoS

def ridgeRegression(X_train, X_test, y_train, y_test):
    ridge = Pipeline([('pca', PCA(.95)), 
                     ('regr', linear_model.Ridge())])
    
    # alphas = [.1, .01, .001, .0001, .00001]
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

    grid_params = [{'regr__alpha' : alphas}]
        
    gridSearch = GridSearchCV(estimator = ridge, param_grid=grid_params, scoring='neg_mean_squared_error', cv = 5, n_jobs=1)
    gridSearch.fit(X_train, y_train)
    
    print("Ridge Regresion:")
    print("Best CV MSE : {}".format(gridSearch.best_score_))
    print("Best params: {}".format(gridSearch.best_params_))
    predictions = gridSearch.predict(X_test)
    print("Test RMSE : {}".format(rmse(predictions, y_test)))
    print()
    
    return gridSearch

def lassoRegression(X_train, X_test, y_train, y_test):
    lasso = Pipeline([('pca', PCA(.95)), 
                     ('regr', linear_model.Lasso())])
    
    # alphas = [.1, .01, .001, .005, .002, .0001, .00001]
    # alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    alphas = [1, 0.1, 0.001, 0.0005]
    grid_params = [{'regr__alpha' : alphas}]
        
    gridSearch = GridSearchCV(estimator = lasso, param_grid=grid_params, scoring='neg_mean_squared_error', cv = 5, n_jobs=1)
    gridSearch.fit(X_train, y_train)
    
    print("Lasso Regresion:")
    print("Best CV MSE : {}".format(gridSearch.best_score_))
    print("Best params: {}".format(gridSearch.best_params_))
    predictions = gridSearch.predict(X_test)
    print("Test RMSE : {}".format(rmse(predictions, y_test)))
    print()
    
    return gridSearch

def elasticNetRegression(X_train, X_test, y_train, y_test):
    elasticNet = Pipeline([('pca', PCA(.95)), 
                     ('regr', linear_model.ElasticNet())])
    
    l1 = [.01, .001, .001, .0001, .1, .5, 1]
    iterations = [1, 5, 10, 20, 50]
    grid_params = [{'regr__l1_ratio' : l1, 'regr__max_iter' : iterations}]
        
    gridSearch = GridSearchCV(estimator = elasticNet, param_grid=grid_params, scoring='neg_mean_squared_error', cv = 5, n_jobs=-1)
    gridSearch.fit(X_train, y_train)
    
    print("ElasticNet Regresion:")
    print("Best CV MSE : {}".format(gridSearch.best_score_))
    print("Best params: {}".format(gridSearch.best_params_))
    predictions = gridSearch.predict(X_test)
    print("Test RMSE : {}".format(rmse(predictions, y_test)))
    print()
    
    return gridSearch


def rfRegression(X_train, X_test, y_train, y_test):
    rf = Pipeline([('pca', PCA(.95)), 
                     ('regr', RandomForestRegressor(n_jobs=-1))])

    n_estimators = [50, 100, 150, 200, 500]
    
    grid_params = [{'regr__n_estimators' : n_estimators}]

    gridSearch = GridSearchCV(estimator = rf, param_grid=grid_params, scoring='neg_mean_squared_error', cv = 5, n_jobs=-1)
    gridSearch.fit(X_train, y_train)
    
    print("RandomForest Regresion:")
    print("Best CV MSE : {}".format(gridSearch.best_score_))
    print("Best params: {}".format(gridSearch.best_params_))
    predictions = gridSearch.predict(X_test)
    print("Test RMSE : {}".format(rmse(predictions, y_test)))
    print()

    return gridSearch

    


def load_data(percent_test):
    train = pd.read_csv('/Users/Tomas/Desktop/Kaggle-House-Prices-Challenge/data/final_train.csv', index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(train, train['SalePrice'], test_size=percent_test, random_state=42)
    X_train = X_train.drop(['SalePrice'], axis=1)
    X_test = X_test.drop(['SalePrice'], axis=1)

    return X_train, X_test, y_train, y_test



def generate_predictions(model, model_name, file_info, test=None):
    if test is None:
        test = pd.read_csv('/Users/Tomas/Desktop/Kaggle-House-Prices-Challenge/data/final_test.csv', index_col=0)
        print(Utilities.summarize_missing(test))
        print(test.shape)

        predictions = model.predict(test)
        predictions = pd.DataFrame(predictions, index = test.index, columns=['SalePrice'])

        predictions.to_csv('/Users/Tomas/Desktop/Kaggle-House-Prices-Challenge/data/{}_{}.csv'.format(model_name, file_info))
    else:
        PATH = "/Users/Tomas/Desktop/Kaggle-House-Prices-Challenge/data/"  #where you put the files
        preds = model.predict(test)
        my_submission = pd.DataFrame({'Id': test.index, 'SalePrice': preds})
        my_submission.to_csv(f'{PATH}submission.csv', index=False)