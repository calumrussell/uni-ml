from common import (random_search_cv_logistic, 
    write_model,
    get_test_set,
    get_train_set)

from sklearn.metrics import brier_score_loss

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
        return brier_score_loss(actual, self.predict(shots))

    def predict(self, shots):
        x = []
        for shot in shots:
            x.append(V1._shot_to_features(shot))
        return [i[1] for i in self.model.predict_proba(x)]

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
            x.append(V1._shot_to_features(shot))

        (model, score) = random_search_cv_logistic(x, y)
        to_obj = V1(model, score*-1, x, y)
        write_model("v1", to_obj)
        return to_obj

if __name__ == "__main__":
    v1 = V1.train()
    print(v1.score)
    print(v1.test_score())
