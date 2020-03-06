#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 07:39:29 2020

@author: grael
"""
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def my_GridSearchCV(
        model_constructor,
        model_kwargs,
        param_grid,
        inner_cv=StratifiedKFold(5)):
    """
    This is a convenience wrapper for GridSearchCV.
    It is actually not the best option when I want to do CV tuning
    on something like LogisticRegression or
    """
    assert type(model_kwargs)==dict
    assert type(param_grid) in [dict, set, list]
    model = model_constructor(**model_kwargs)
    # I use ROC AUC for parameter tuning, specified here.
    if param_grid != {}:
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring='balanced_accuracy',
            refit=True,
            cv=inner_cv)
    else:
        grid_search = model
    return grid_search


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


xgb_kwargs = {
    'n_estimators': 500,
    'class_weight': 'balanced',
    'subsample': 0.8,}
xgb_tuning = {
    'max_depth': [],
    'min_child_weight': [],
    'gamma': []}
xgb = Pipeline(
    [('classification', my_GridSearchCV(
          XGBClassifier,
          xgb_kwargs, xgb_tuning))])