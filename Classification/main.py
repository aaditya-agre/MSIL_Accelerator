import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from models.trainer import get_model
from utils.metrics import classification_metrics
from utils.save_model import save_model
from utils.data_split import train_test_split_data
from tuning.hyperparam import hyperparams, visualize_results, logger
import joblib

# Load classification dataset
df = pd.read_csv('C:\\Users\\HP\\Downloads\\Iris.csv')

# Use a classification target
X = df.drop('Species', axis=1)
y = df['Species']

# Split the data
X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42, stratify=y)

# Get model
model = get_model('random_forest',task='classification')

# Get hyperparameter grid
param_grid = hyperparams(str(model))

# Tuning function
def tune_model(model, param_grid, X_train, y_train, scoring='accuracy', operation='GridSearchCV', cv=5):
    if operation == 'GridSearchCV':
        grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_, grid
    
    elif operation == 'RandomizedSearchCV':
        random_search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=20,
            scoring=scoring,
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_, random_search.best_params_, random_search
    
    elif operation == 'None':
        return model, model.get_params(), None

# Tune model
tuned_model, best_params, tuned_obj = tune_model(model, param_grid, X_train, y_train, operation='RandomizedSearchCV')

# Visualize results
visualize_results(tuned_obj, param_name='n_estimators')  # Change this param based on model

# Log best params
logger(str(tuned_model), best_params)

# Train on full training data
model = tuned_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
metrics = classification_metrics(y_test, y_pred, average='macro')
print("Evaluation Metrics:", metrics)

# Save model (optional)
joblib.dump(tuned_model, 'random_forest_model_tuned.pkl')
print("Model saved as 'random_forest_model_tuned.pkl'")

# Load model and predict (optional)
# tuned_model = joblib.load('random_forest_model_tuned.pkl')
# tuned_model.predict(X_test[:5])
