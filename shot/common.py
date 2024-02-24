import time
import pickle
import os
import json

from sklearn.metrics import make_scorer, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

class ShotFeatures:
    """
    Categorical features used in shot prediction model.

    location - region of the opponent's half from which the shot was taken
    play - type of play within which the shot was taken
    body_part - part of shooter's body used to take the shot
    """

    location = [
        'BoxCentre',
        'BoxLeft',
        'BoxRight',
        'DeepBoxLeft',
        'DeepBoxRight',
        'OutOfBoxCentre',
        'OutOfBoxDeepLeft',
        'OutOfBoxDeepRight',
        'OutOfBoxLeft',
        'OutOfBoxRight',
        'SmallBoxCentre',
        'SmallBoxLeft',
        'SmallBoxRight',
        'ThirtyFivePlusCentre',
        'ThirtyFivePlusLeft',
        'ThirtyFivePlusRight'] 

    play = [
        'DirectFreekick',
        'FastBreak',
        'FromCorner',
        'Penalty',
        'RegularPlay',
        'SetPiece',
        'ThrowinSetPiece']

    body_part = [
        'Head',
        'LeftFoot',
        'OtherBodyPart',
        'RightFoot']


def make_shot_scorer():
    """
    Almost all predictions will come back as miss and skew our score. Although we are technically
    predicting a class, we are looking for a measure of distance (i.e. a model that predicts 55% 
    is better than one predicts 40% for a shot that goes in, even though both are wrong the first 
    more accurate).

    Brier score is equivalent to MSE for classification.
    """
    return make_scorer(score_func=brier_score_loss, greater_is_better=False, response_method='predict_proba')

def write_model(model_version, model_obj):
    epoch = round(time.time())
    
    with open(f"shot/models/{model_version}_{epoch}.pkl", 'wb') as f:
        pickle.dump(model_obj, f)
    return

def get_train_set():
    path = 'data/train_shots.json'
    shots = []
    with open(path, 'r') as f:
        for line in f.readlines():
            shots.append(json.loads(line))
    return shots

def get_test_set():
    path = 'data/test_shots.json'
    shots = []
    with open(path, 'r') as f:
        for line in f.readlines():
            shots.append(json.loads(line))
    return shots

def get_model_best_score():
    last_score = 1
    best_model = None
    models = os.listdir('shot/models')
    for model_path in models:
        path, extension = model_path.split(".")
        version, epoch = path.split("_")
        with open(f"shot/models/{model_path}", "rb") as f:
            tmp_model = pickle.load(f)
            score = tmp_model.test_score()
            if float(score) < last_score:
                best_model = tmp_model
                last_score = score
    return best_model

def random_search_cv_xgb(x, y):
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    xgb_model = xgb.XGBClassifier(tree_method='hist')
    random_search = RandomizedSearchCV(
            xgb_model, 
            param_distributions=params, 
            scoring=make_shot_scorer(), 
            n_iter=20, 
            cv=10, 
    )

    random_search.fit(x,y)
    return (random_search.best_estimator_, random_search.best_score_)

def random_search_cv_logistic(x, y):
    params = {
        'class_weight': [
            {0: 0.5, 1: 0.5},
            {0: 0.75, 1: 0.25},
            {0: 0.25, 1: 0.75},
            {0: 0.6, 1: 0.4},
            {0: 0.4, 1: 0.6},
            None,
        ],
        'solver': ['lbfgs', 'liblinear', 'newton-cholesky', 'newton-cg'],
        'penalty': ['l2', None],
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    logistic_model = LogisticRegression(max_iter=2000)
    random_search = RandomizedSearchCV(
            logistic_model,
            param_distributions=params,
            scoring=make_shot_scorer(),
            n_iter=20,
            cv=10,
    )
    random_search.fit(x,y)
    return (random_search.best_estimator_, random_search.best_score_)
