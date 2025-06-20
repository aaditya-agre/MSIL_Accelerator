from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

def feature_engineering(X, method=None, **kwargs):
    if method == 'poly':
        degree = kwargs.get('degree', 2)
        return PolynomialFeatures(degree=degree).fit_transform(X)
    elif method == 'pca':
        return PCA(n_components=kwargs.get('n_components', 2)).fit_transform(X)
    elif method == 'selectkbest':
        k = kwargs.get('k', 5)
        return SelectKBest(score_func=f_regression, k=k).fit_transform(X, kwargs.get('y'))
    return X