import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from models.trainer import get_model
from utils.metrics import regression_metrics
from utils.save_model import save_model
from utils.data_split import train_test_split_data
from tuning.hyperparam import hyperparams, visualize_results, logger
import joblib

df = pd.read_csv('C:\\Users\\Ashok Kumar\\Desktop\\iris.csv')
X = df.drop('SepalLengthCm', axis=1)
y = df['SepalLengthCm']

X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)

model = get_model('ridge', task='regression')

param_grid = hyperparams(str(model))

def tune_model(model, param_grid, X_train, y_train, scoring='r2', operation = 'GridSeachCV',cv=5):
    if operation == 'GridSearchCV':
        grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_, grid
    if operation == 'RandomizedSearchCV':
        random_search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=20,                # Number of parameter settings sampled
            scoring=scoring,             # Evaluation metric
            cv=cv,                     # 5-fold cross-validation
            verbose=1,
            random_state=42,
            n_jobs=-1
            )
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_, random_search.best_params_, random_search
    if operation == 'None':
        return model, model.get_params(), None

tuned_model, best_params, tuned_obj = tune_model(model, param_grid, X_train, y_train, operation = 'RandomizedSearchCV')
visualize_results(tuned_obj, param_name = 'alpha')
logger(str(tuned_model), best_params)
print("Best Hyperparameters:", best_params)
model = tuned_model.fit(X_train, y_train)

y_pred = model.predict(X_test)
metrics = regression_metrics(y_test, y_pred)
print("Evaluation Metrics:", metrics)



#joblib.dump((tuned_model), 'ridge_model_tuned.pkl')
#print("Model and transformer saved as 'ridge_model_tuned.pkl'")

# Load the model and make predictions
#tuned_model = joblib.load('ridge_model_tuned.pkl')
#tuned_model.predict(X_test[:5])