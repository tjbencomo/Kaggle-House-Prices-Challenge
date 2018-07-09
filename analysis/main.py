import Models, Utilities


def main():
    X_train, X_test, y_train, y_test = Models.load_data(.5)

   los =  Models.linearRegression(X_train, X_test, y_train, y_test)
    ridge = Models.ridgeRegression(X_train, X_test, y_train, y_test)
    Models.lassoRegression(X_train, X_test, y_train, y_test)
    model = Models.elasticNetRegression(X_train, X_test, y_train, y_test)

    Models.generate_predictions(model, "50_percent_test_data")


if __name__ == "__main__": 
    main()