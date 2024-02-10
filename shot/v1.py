import json
import pickle
import time
import os

from common import random_search_cv_logistic

class V1:
    """
    Basic model used for testing:
        * Distance
        * Angle
        * Distance * Angle
    """
    def __init__(self, model, score, x, y):
        self.model = model
        self.score = score
        self.x = x
        self.y = y

    @staticmethod
    def best_score():
        last_score = 0
        best_model = None
        models = os.listdir('shot/models')
        for model_path in models:
            with open(f"shot/models/{model_path}", "rb") as f:
                tmp_model = pickle.load(f)
                if tmp_model.score > last_score:
                    best_model = tmp_model
        return best_model

    @staticmethod
    def most_recent():
        last = -1
        most_recent_model_path = None

        models = os.listdir('shot/models')
        for model in models:
            path, extension = model.split(".")
            version, epoch = path.split("_")
            if version == "v1" and int(epoch) > last:
                last = epoch
                most_recent_model_path = model

        if most_recent_model_path:
            with open(f"shot/models/{most_recent_model_path}", "rb") as f:
                return pickle.load(f)
        return

    @staticmethod
    def train():
        """
        Train should return a copy of self with the actual completed model. So train is an object
        creator.
        """

        path = 'data/train_shots.json'

        x = []
        y = []

        shots = []

        with open(path, 'r') as f:
            for line in f.readlines():
                shots.append(json.loads(line))

        for shot in shots:
            y.append(shot["result"])
            distance = shot["distance"]
            angle = shot["angle"]
            x.append([distance, angle, distance * angle])

        (model, score) = random_search_cv_logistic(x, y)
        to_obj = V1(model, score, x, y)
        epoch = round(time.time())
        
        with open(f"shot/models/v1_{epoch}.pkl", 'wb') as f:
            pickle.dump(to_obj, f)
        return to_obj


if __name__ == "__main__":

    v1 = V1.best_score()
    print(v1.score)
