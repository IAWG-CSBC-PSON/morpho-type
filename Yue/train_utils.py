'''
From Elliot Gray
Modified by Yue Qin
'''

'''
# Elastic Net
enet_kwargs = {
        'solver': 'saga',
        'penalty': 'elasticnet',
        'max_iter': 10000,
        }
enet_tuning = {
        'l1_ratio': [.1, .7, .95],
        'C': [.01, .1, .5]
        }
enet = Pipeline(
    [('standardization', StandardScaler()),
     ('classification', my_GridSearchCV(
        LogisticRegression,
        enet_kwargs, enet_tuning))])
# Random Forest
rf_kwargs = {
    'n_estimators': 200,
    'class_weight': 'balanced'}
rf_tuning = {
    'min_samples_split': [.05, .1, .2],
    'max_features': ['sqrt', .5, .75]}
rf = Pipeline(
    [('classification', my_GridSearchCV(
         RandomForestClassifier,
         rf_kwargs, rf_tuning))])
# Extremely randomized trees
et_kwargs = {
    'n_estimators': 200,
    'class_weight': 'balanced'}
et_tuning = {
    'min_samples_split': [.05, .1, .2],
    'max_features': ['sqrt', .5, .75]}
et = Pipeline(
    [('classification', my_GridSearchCV(
         ExtraTreesClassifier,
         et_kwargs, et_tuning))])
'''

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer

def spearman_score(true_probs, pred_probs):
    """
    A metric to assess multilabel regression performance using the 
    un-weighted class-average 0-clipped Spearman Correlation.
    can be passes to sklearn make_scorer
    """
    m, c = true_probs.shape
    assert m == pred_probs.shape[0]
    if true_probs.ndim == 1:
        s, p = spearmanr(true_probs, pred_probs)
    elif true_probs.ndim == 2:
        assert c == pred_probs.shape[1]
        s_list = []
        for n in range(c):
            s, p = spearmanr(true_probs[:,n], pred_probs[:,n])
            s_list.append(s)
        s = np.mean(np.array(s_list))
    else:
        raise ValueError
    # don't allow negative correlations to be rewarded.
    s = np.clip(s, 0., 1.)
    return s

scorer = make_scorer(spearman_score, needs_proba=True)

def my_GridSearchCV(
        model_constructor,
        model_kwargs,
        param_grid,
        scoring='balanced_accuracy',
        inner_cv=StratifiedKFold(n_splits=5, random_state=42)):
    """
    This is a convenience wrapper for GridSearchCV.
    It is actually not the best option when I want to do CV tuning
    on something like LogisticRegression or
    """
    assert type(model_kwargs) == dict
    assert type(param_grid) in [dict, set, list]
    model = model_constructor(**model_kwargs)
    # I use ROC AUC for parameter tuning, specified here.
    if param_grid != {}:
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring=scoring,
            refit=True,
            cv=inner_cv)
    else:
        grid_search = model
    return grid_search

