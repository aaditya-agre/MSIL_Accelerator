import matplotlib.pyplot as plt
import pandas as pd

def hyperparams(model):
    if model == 'LinearRegression()':
        param_grid = {
    'copy_X': [True, False],
    'fit_intercept': [True, False],
    'n_jobs': [None, -1, 1],  
    'positive': [True, False],
    'tol': [1e-6, 1e-4, 1e-2, 1e-1, 1.0],
    }
        return param_grid
    
    if model == 'Ridge()':
        param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],         # Regularization strength
    'copy_X': [True, False],                        # Whether to copy input X
    'fit_intercept': [True, False],                 # Whether to fit intercept
    'max_iter': [None, 100, 500, 1000, 5000],        # Max number of iterations for solvers
    'positive': [False],                      # Force coefficients to be positive
    'random_state': [None, 42],                     # For reproducibility
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],  # Optimization algorithms
    'tol': [1e-6, 1e-4, 1e-2, 1e-1]                 # Convergence tolerance
    }

        return param_grid
    
    if model == 'Lasso()':
        param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],         # Regularization strength
    'copy_X': [True, False],                               # Whether to copy input X
    'fit_intercept': [True, False],                        # Whether to fit intercept
    'max_iter': [500, 1000, 5000, 10000],                  # Maximum number of iterations
    'positive': [True, False],                             # Force coefficients to be positive
    'precompute': [False, True],                           # Whether to use Gram matrix (faster for small datasets)
    'random_state': [None, 42],                            # For reproducibility
    'selection': ['cyclic', 'random'],                     # Coordinate descent strategy
    'tol': [1e-6, 1e-4, 1e-2, 1e-1],                       # Tolerance for stopping criterion
    'warm_start': [False, True]                            # Use solution of previous call to initialize
    }

        return param_grid
    
    if model == 'DecisionTreeRegressor()':
        param_grid = {
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1],                  # Complexity parameter for Minimal Cost-Complexity Pruning
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],  # Split quality metrics
    'max_depth': [None, 5, 10, 20, 50],                    # Max depth of the tree
    'max_features': [None, 'auto', 'sqrt', 'log2'],        # Number of features to consider at each split
    'max_leaf_nodes': [None, 10, 20, 50, 100],             # Max number of leaf nodes
    'min_impurity_decrease': [0.0, 0.001, 0.01],           # Node split threshold
    'min_samples_leaf': [1, 2, 4, 10],                     # Minimum number of samples required at a leaf node
    'min_samples_split': [2, 5, 10],                       # Minimum number of samples required to split a node
    #'min_weight_fraction_leaf': [0.0, 0.01, 0.1],          # Minimum weighted fraction of the input samples at a leaf node
    #'monotonic_cst': [None],                               # Only used for monotonic regression tasks (keep None unless using it)
    'random_state': [None, 42],                            # For reproducibility
    #'splitter': ['best', 'random']                         # Strategy to choose split at each node
    }

        return param_grid
    
    if model == 'RandomForestRegressor()':
        param_grid = {
    'n_estimators': [50, 100, 200, 500],                     # Number of trees in the forest
    'criterion': ['squared_error', 'absolute_error', 'poisson', 'friedman_mse'],  # Split quality function
    'max_depth': [None, 10, 20, 50],                         # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],                         # Min samples to split an internal node
    'min_samples_leaf': [1, 2, 4],                           # Min samples required at a leaf node
    'min_weight_fraction_leaf': [0.0, 0.01, 0.1],            # Min weighted fraction of samples at leaf
    'max_features': ['auto', 'sqrt', 'log2', 0.5, 1.0],      # Number of features to consider at each split
    'max_leaf_nodes': [None, 10, 20, 50],                    # Max number of leaf nodes
    'min_impurity_decrease': [0.0, 0.001, 0.01],             # Split if gain is larger than this
    'bootstrap': [True, False],                              # Use bootstrap samples
    'oob_score': [False, True],                              # Use out-of-bag samples to estimate score
    'max_samples': [None, 0.5, 0.8],                         # If bootstrap=True, number of samples to draw
    'ccp_alpha': [0.0, 0.01, 0.1],                           # Complexity parameter for Minimal Cost-Complexity Pruning
    'n_jobs': [-1],                                          # Use all processors
    'random_state': [None, 42],                              # For reproducibility
    'verbose': [0],                                          # Verbosity (usually 0 in GridSearch)
    'warm_start': [False, True],                             # Use previous fit results to add more trees
    'monotonic_cst': [None]                                  # Only used for monotonic constraints (optional / experimental)
    }

        return param_grid
    
    if model == 'SVR()':
        param_grid = {
    'C': [0.1, 1.0, 10.0, 100.0],                          # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],        # Kernel type
    'degree': [2, 3, 4, 5],                                # Degree for 'poly' kernel
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],       # Kernel coefficient
    'coef0': [0.0, 0.1, 0.5, 1.0],                         # Independent term for 'poly' and 'sigmoid' kernels
    'epsilon': [0.01, 0.1, 0.2, 0.5],                      # Epsilon-tube within which no penalty is associated
    'shrinking': [True, False],                            # Use shrinking heuristic
    'tol': [1e-4, 1e-3, 1e-2],                             # Tolerance for stopping criterion
    'cache_size': [100, 200, 500],                         # Size of kernel cache (in MB)
    'max_iter': [-1, 1000, 5000],                          # Max iterations; -1 = no limit
    'verbose': [False]                                     # Verbosity (usually keep False in GridSearch)
    }

        return param_grid
    
    if model == 'xgb.XGBRegressor()':
        param_grid = {
    'n_estimators': [100, 200, 500],                     # Number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1, 0.3],             # Step size shrinkage
    'max_depth': [3, 5, 7, 10],                          # Max depth of each tree
    'min_child_weight': [1, 3, 5],                       # Min sum of instance weight needed in a child
    'gamma': [0, 0.1, 0.5, 1],                           # Minimum loss reduction to make a split
    'subsample': [0.6, 0.8, 1.0],                        # Fraction of samples used per tree
    'colsample_bytree': [0.6, 0.8, 1.0],                 # Fraction of features used per tree
    'reg_alpha': [0, 0.1, 1.0],                          # L1 regularization term
    'reg_lambda': [1.0, 1.5, 2.0],                       # L2 regularization term
    'booster': ['gbtree', 'gblinear', 'dart'],          # Type of boosting model
    'tree_method': ['auto', 'exact', 'approx', 'hist'], # Tree construction method
    'objective': ['reg:squarederror'],                  # Loss function (regression default)
    'verbosity': [0],                                    # Silent output
    'random_state': [42]                                 # For reproducibility
    }
        
def visualize_results(search_obj, param_name, buffer):
    """
    search_obj: fitted GridSearchCV or RandomizedSearchCV object
    param_name: parameter to plot (must be in search_obj.param_grid or param_distributions)
    """
    results = pd.DataFrame(search_obj.cv_results_)
    if param_name not in results.columns and f'param_{param_name}' in results.columns:
        param_name = f'param_{param_name}'

    if param_name not in results.columns:
        raise ValueError(f"Parameter '{param_name}' not found in search results.")

    plt.figure(figsize=(8, 5))
    plt.plot(results[param_name], results['mean_test_score'], marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Mean CV Score')
    plt.title(f'Performance vs {param_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buffer, format = 'png')
    
import json

def logger(model_name: str, best_params, best_score=None, save_path=None):
    log = {
        'model': model_name,
        'best_params': best_params
    }
    if best_score is not None:
        log['best_score'] = best_score

    print(f"Tuning Summary for {model_name}")
    print(json.dumps(log, indent=4))

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(log, f, indent=4)
        print(f"\n Log saved to: {save_path}")



    
    
    

    
    