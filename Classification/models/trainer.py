from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier,
    AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

def get_model(name, task='classification'):
    models = {
        'logistic': LogisticRegression(),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(),
        'gradient_boosting': GradientBoostingClassifier(),
        'xgb': xgb.XGBClassifier(),
        'gaussian_nb': GaussianNB(),
        'multinomial_nb': MultinomialNB(),
        'bernoulli_nb': BernoulliNB(),
        'linear_svm': SVC(kernel='linear'),
        'kernel_svm': SVC(kernel='rbf'),
        'voting': VotingClassifier(estimators=[
            ('lr', LogisticRegression()), 
            ('rf', RandomForestClassifier()), 
            ('svc', SVC(probability=True))
        ], voting='soft'),
        'bagging': BaggingClassifier(estimator=DecisionTreeClassifier()),
        'adaboost': AdaBoostClassifier(estimator=DecisionTreeClassifier()),
        'ann': MLPClassifier(max_iter=1000)
    }
    return models.get(name)
