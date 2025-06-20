
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

def get_model(name, task='regression'):
      models = {
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'decision_tree': DecisionTreeRegressor(),
            'random_forest': RandomForestRegressor(),
            'svr': SVR(),
            'xgb': xgb.XGBRegressor()
      }
      return models.get(name)