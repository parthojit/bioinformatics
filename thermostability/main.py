import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

class ThermoClassifer(object):
    def __init__(self):
        self.df_train = pd.read_csv('../thermostability/src/ddg_bin_train.csv')
        self.df_test = pd.read_csv('../thermostability/src/ddg_bin_test.csv')

    def return_array(self, df:pd.DataFrame):
        X = df.iloc[:,0:len(df.columns)-1]
        y = pd.get_dummies(df['ddGC'])
        y = y['stable']
        return X,y
    
    def fetch_data(self):
        X_train, y_train = self.return_array(self.df_train)
        X_test, y_test = self.return_array(self.df_test)
        return X_train, X_test, y_train, y_test

    def svm_params(self, X_train, X_test, y_train, y_test):
        param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['scale', 'auto'],
              'kernel': ['linear']}
        grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, n_jobs=-1)
        grid.fit(X_train, y_train)
        print(grid.best_params_) 
        grid_predictions = grid.predict(X_test) 
        # print classification report 
        print(classification_report(y_test, grid_predictions)) 

    def ann_params(self, X_train, X_test, y_train, y_test):
        param_grid = {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }
        grid = GridSearchCV(MLPClassifier(), param_grid, refit = True, verbose = 3, n_jobs=-1)
        grid.fit(X_train, y_train)
        print(grid.best_params_) 
        grid_predictions = grid.predict(X_test) 
        # print classification report 
        print(classification_report(y_test, grid_predictions)) 


    def run(self):
        X_train, X_test, y_train, y_test = self.fetch_data()
        # self.svm_params(X_train, X_test, y_train, y_test)
        self.ann_params(X_train, X_test, y_train, y_test)

if __name__=="__main__":
    t = ThermoClassifer()
    t.run()