import os

from common import (random_search_cv_logistic, 
    get_model_best_score, 
    get_model_most_recent, 
    write_model,
    get_test_set,
    get_train_set)
from sklearn.metrics import roc_auc_score

class V1:
    """
    Basic model used for testing:
        * distance
        * angle
        * distance * angle
    """
    def __init__(self, model, score, x, y):
        self.model = model
        self.score = score
        self.x = x
        self.y = y

    @staticmethod
    def _shot_to_features(shot):
         distance = shot["distance"]
         angle = shot["angle"]
         return [distance, angle, distance*angle]

    def test_score(self):
        shots = get_test_set()
        actual = []
        for shot in shots:
            actual.append(shot["result"])

        # We just want probability of goal
        probs = [i[1] for i in self.predict(shots)]
        return roc_auc_score(actual, probs)

    def predict(self, shots):
        x = []
        for shot in shots:
            x.append(self._shot_to_features(shot))
        return self.model.predict_proba(x)

    @staticmethod
    def most_recent():
        return get_model_most_recent("v1")

    @staticmethod
    def best_score():
        return get_model_best_score("v1")

    @staticmethod
    def train():
        """
        Train should return a copy of self with the actual completed model. So train is an object
        creator.
        """
        x = []
        y = []
        shots = get_train_set()

        for shot in shots:
            y.append(shot["result"])
            x.append(self._shot_to_features(shot))

        (model, score) = random_search_cv_logistic(x, y)
        to_obj = V1(model, score, x, y)
        write_model("v1", to_obj)
        return to_obj

if __name__ == "__main__":

    v1 = V1.best_score()
    print(v1.score)
    print(v1.test_score())