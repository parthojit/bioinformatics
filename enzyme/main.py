from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

class Enzyme(object):
    def __init__(self):
        self.fetch_data()

    def fetch_data(self):
        with open('../enzyme/src/clf/X_train.pkl','rb') as f:
            X_train = pickle.load(f)
        with open('../enzyme/src/clf/X_test.pkl','rb') as f:
            X_test = pickle.load(f)
        with open('../enzyme/src/clf/y_train.pkl','rb') as f:
            y_train = pickle.load(f)
        with open('../enzyme/src/clf/y_test.pkl','rb') as f:
            y_test = pickle.load(f)
        return X_train, X_test, y_train, y_test

    def svm_params(self, X_train, X_test, y_train, y_test):
        param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['scale', 'auto'],
              'kernel': ['rbf']}
        grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, n_jobs=-1, cv=10)
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
        grid = GridSearchCV(MLPClassifier(), param_grid, refit = True, verbose = 3, n_jobs=-1, cv=10)
        grid.fit(X_train, y_train)
        print(grid.best_params_) 
        grid_predictions = grid.predict(X_test) 
        # print classification report 
        print(classification_report(y_test, grid_predictions))
    
    def rf_params(self, X_train, X_test, y_train, y_test):
        param_grid = {
            'bootstrap': [True],
            'min_samples_leaf': [5],
            'min_samples_split': [2],
            'n_estimators': [100, 200, 300, 1000]
        }
        grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 3, n_jobs=-1, cv=10)
        grid.fit(X_train, y_train)
        print(grid.best_params_) 
        grid_predictions = grid.predict(X_test) 
        # print classification report 
        print(classification_report(y_test, grid_predictions))

    def nb_params(self, X_train, X_test, y_train, y_test):
        pass

    def fnn_params(self, X_train, X_test, y_train, y_test):
        pass

    def run(self):
        X_train, X_test, y_train, y_test = self.fetch_data()
        self.rf_params(X_train, X_test, y_train, y_test)
        # self.ann_params(X_train, X_test, y_train, y_test)

if __name__=="__main__":
    e = Enzyme()
    e.run()