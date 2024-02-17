import time
import pickle
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

class ShotFeatures:
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
    last_score = 0
    best_model = None
    models = os.listdir('shot/models')
    for model_path in models:
        path, extension = model_path.split(".")
        version, epoch = path.split("_")
        with open(f"shot/models/{model_path}", "rb") as f:
            tmp_model = pickle.load(f)
            if float(tmp_model.score) > last_score:
                best_model = tmp_model
                last_score = tmp_model.score
    return best_model

def get_model_most_recent():
    last = -1
    most_recent_model_path = None

    models = os.listdir('shot/models')
    for model_path in models:
        path, extension = model_path.split(".")
        version, epoch = path.split("_")
        if int(epoch) > int(last):
            last = epoch
            most_recent_model_path = model_path

    if most_recent_model_path:
        with open(f"shot/models/{most_recent_model_path}", "rb") as f:
            return pickle.load(f)
    return

def random_search_cv_logistic(x, y):
    params = {
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'penalty': ['l2'],
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    logistic_model = LogisticRegression(max_iter=2000)
    random_search = RandomizedSearchCV(
            logistic_model,
            param_distributions=params,
            scoring='roc_auc',
            n_jobs=4,
            cv=5,
    )
    random_search.fit(x,y)
    return (random_search.best_estimator_, random_search.best_score_)
