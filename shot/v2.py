import os

from common import (random_search_cv_logistic, 
    get_model_best_score, 
    get_model_most_recent, 
    write_model,
    get_test_set,
    get_train_set,
    ShotFeatures)

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

class V2:
    """
    Basic model used for testing LabelBinarizer:
        * distance
        * angle
        * distance * angle
        * body_part
    """
    def __init__(self, model, score, x, y, encoder):
        self.model = model
        self.score = score
        self.x = x
        self.y = y
        self.encoder = encoder

    @staticmethod
    def _shots_to_features(shots, encoder):
        normal = []
        for shot in shots:
            distance = shot["distance"]
            angle = shot["angle"]
            normal.append([distance, angle, distance*angle])

        one_hot = []
        for shot in shots:
            one_hot.append([shot["body_part"]])
        encoded = encoder.transform(one_hot).toarray()

        return [[*i, *j] for i, j in zip(normal, encoded)]

    def test_score(self):
        shots = get_test_set()
        actual = []
        for shot in shots:
            actual.append(shot["result"])

        # We just want probability of goal
        probs = [i[1] for i in self.predict(shots)]
        return roc_auc_score(actual, probs)

    def predict(self, shots):
        return self.model.predict_proba(self._shots_to_features(shots, self.encoder))

    @staticmethod
    def most_recent():
        return get_model_most_recent("v2")

    @staticmethod
    def best_score():
        return get_model_best_score("v2")

    @staticmethod
    def train():
        """
        Train should return a copy of self with the actual completed model. So train is an object
        creator.
        """
        x = []
        y = []
        shots = get_train_set()

        encoder = OneHotEncoder(categories=[ShotFeatures.body_part])

        encoder_data = []
        for shot in shots:
            encoder_data.append([shot["body_part"]])
        encoder.fit(encoder_data)
            
        for shot in shots:
            y.append(shot["result"])
        x = V2._shots_to_features(shots, encoder)

        (model, score) = random_search_cv_logistic(x, y)
        to_obj = V2(model, score, x, y, encoder)
        write_model("v2", to_obj)
        return to_obj

if __name__ == "__main__":

    v2 = V2.train()
    print(v2.score)
    print(v2.test_score())