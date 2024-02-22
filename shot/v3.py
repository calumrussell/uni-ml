from sklearn.metrics import mean_squared_error
from common import (random_search_cv_logistic, 
    write_model,
    get_test_set,
    get_train_set,
    ShotFeatures)

from sklearn.preprocessing import OneHotEncoder

class V3:
    """
    Full featured model:
        * distance
        * angle
        * distance * angle
        * body_part
        * shot_location
        * shot_play
        * big_chance
        * fast_break
        * assisted
        * first_touch
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
            normal.append([
                distance, 
                angle, 
                distance*angle,
                shot["big_chance"],
                shot["fast_break"],
                shot["assisted"],
                shot["first_touch"],
            ])

        one_hot = []
        for shot in shots:
            one_hot.append([shot["shot_location"], shot["body_part"], shot["shot_play"]])
        encoded = encoder.transform(one_hot).toarray()
        return [[*i, *j] for i, j in zip(normal, encoded)]

    def test_score(self):
        shots = get_test_set()
        actual = []
        for shot in shots:
            actual.append(shot["result"])

        # We just want probability of goal
        probs = [i[1] for i in self.predict(shots)]
        return mean_squared_error(actual, probs)

    def predict(self, shots):
        return self.model.predict_proba(self._shots_to_features(shots, self.encoder))

    @staticmethod
    def train():
        """
        Train should return a copy of self with the actual completed model. So train is an object
        creator.
        """
        x = []
        y = []
        shots = get_train_set()

        encoder = OneHotEncoder(categories=[ShotFeatures.location, ShotFeatures.body_part, ShotFeatures.play])

        encoder_data = []
        for shot in shots:
            encoder_data.append([shot["shot_location"], shot["body_part"], shot["shot_play"]])
        encoder.fit(encoder_data)
            
        for shot in shots:
            y.append(shot["result"])
        x = V3._shots_to_features(shots, encoder)

        (model, score) = random_search_cv_logistic(x, y)
        to_obj = V3(model, score, x, y, encoder)
        write_model("v3", to_obj)
        return to_obj
