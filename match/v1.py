from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer

from common import make_brier_multi_scorer_with_lb, write_model, PoissonRatings, brier_multi

def random_search_cv_logistic(x, y):

    params = {
        'solver': ['lbfgs', 'newton-cg'],
        'penalty': ['l2', None],
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    logistic_model = LogisticRegression(multi_class="multinomial")
    lb = LabelBinarizer()
    lb.fit_transform(y)

    random_search = RandomizedSearchCV(
            logistic_model,
            param_distributions=params,
            scoring=make_brier_multi_scorer_with_lb(lb=lb),
            n_iter=20,
            cv=10,
    )
    random_search.fit(x,y)
    
    return (random_search.best_estimator_, random_search.best_score_)

class V1:
    """
    Multiclass logistic classifier that uses difference of team goal rating as param.
    """
    def __init__(self, model, score, ratings_window_length):
        self.model = model
        self.score = score
        self.ratings_window_length = ratings_window_length

    def predict(self, home_rating, away_rating):
        rating_diff = home_rating - away_rating
        pred = self.model.predict_proba([[rating_diff]])
        return {i: j for i, j in zip(self.model.classes_, pred[0])}

    def test_score(self):
        ratings_model = PoissonRatings.test(self.ratings_window_length)
        ratings = ratings_model.goal_ratings
        x = []
        actuals = []

        for match in ratings:
            rating = ratings[match]

            home_rating = rating[0]
            away_rating = rating[1]
            home_goal_diff = rating[2]

            win = 0
            draw = 0
            loss = 0

            if home_goal_diff > 0:
                win = 1
            elif home_goal_diff == 0:
                draw = 1
            else:
                loss = 1
            x.append([home_rating - away_rating])
            actuals.append([win, draw, loss])

        preds = []
        probs = self.model.predict_proba(x)
        for prob in probs:
            win = 0
            draw = 0
            loss = 0
            for outcome, p in zip(self.model.classes_, prob):
                if outcome == "win":
                    win = p
                elif outcome == "draw":
                    draw = p
                else:
                    loss = p
            preds.append([win, draw, loss])
        return brier_multi(actuals, preds) * -1

    @staticmethod
    def train(ratings_window_length):
        ratings_model = PoissonRatings.train(ratings_window_length)
        ratings = ratings_model.goal_ratings

        x = []
        y = []

        for match in ratings:
            rating = ratings[match]

            home_rating = rating[0]
            away_rating = rating[1]
            home_goal_diff = rating[2]

            result = "loss"
            if home_goal_diff > 0:
                result = "win"
            elif home_goal_diff == 0:
                result = "draw"
            x.append([home_rating - away_rating])
            y.append(result)

        (model, score) = random_search_cv_logistic(x, y)
        obj = V1(model, score, ratings_window_length)
        write_model("v1", obj)
        return obj

