import joblib
import os

def save_model(model, model_name, path='artifacts/'):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, f"{model_name}.pkl"))
    return os.path.join(path, f"{model_name}.pkl")