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
