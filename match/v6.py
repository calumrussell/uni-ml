from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer
import xgboost as xgb

from common import PoissonRatings, make_brier_multi_scorer_with_lb, write_model, brier_multi

def random_search_cv_xgb(x, y):
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    xgb_model = xgb.XGBClassifier(tree_method='hist', objective='multi:softprob', num_class=3)
    lb = LabelBinarizer()
    lb.fit_transform(y)

    random_search = RandomizedSearchCV(
            xgb_model, 
            param_distributions=params, 
            scoring=make_brier_multi_scorer_with_lb(lb=lb),
            n_iter=20, 
            cv=10, 
    )

    random_search.fit(x,y)
    return (random_search.best_estimator_, random_search.best_score_)

class V6:
    """
    Multiclass XGBoost classifier that uses difference of team goal rating and difference of
    expected goal rating.
    """
    def __init__(self, model, score, ratings_window_length):
        self.model = model
        self.score = score
        self.ratings_window_length = ratings_window_length

    def predict(self, home_rating, away_rating):
        rating_diff = home_rating - away_rating
        pred = self.model.predict_proba([[rating_diff]])
        print(pred)
        return {i: j for i, j in zip(self.model.classes_, pred[0])}

    def test_score(self):
        ratings_model = PoissonRatings.test(self.ratings_window_length)
        goal_ratings = ratings_model.goal_ratings
        expected_goals_ratings = ratings_model.expected_goals_ratings

        x = []
        actuals = []

        for match in goal_ratings:
            goal_rating = goal_ratings[match]
            expected_goal_rating = expected_goals_ratings[match]

            home_rating = min(goal_rating[0], 4)
            away_rating = min(goal_rating[1], 4)
            home_goal_diff = goal_rating[2]

            x_home_rating = expected_goal_rating[0]
            x_away_rating = expected_goal_rating[1]

            win = 0
            draw = 0
            loss = 0

            if home_goal_diff > 0:
                win = 1
            elif home_goal_diff == 0:
                draw = 1
            else:
                loss = 1
            x.append([home_rating - away_rating, x_home_rating - x_away_rating])
            actuals.append([win, draw, loss])

        preds = []
        probs = self.model.predict_proba(x)
        for prob in probs:
            win = 0
            draw = 0
            loss = 0
            for outcome, p in zip(self.model.classes_, prob):
                if outcome == 0:
                    win = p
                elif outcome == 1:
                    draw = p
                else:
                    loss = p
            preds.append([win, draw, loss])

        return brier_multi(actuals, preds) * -1

    @staticmethod
    def train(ratings_window_length):
        ratings_model = PoissonRatings.train(ratings_window_length)
        goal_ratings = ratings_model.goal_ratings
        expected_goals_ratings = ratings_model.expected_goals_ratings

        x = []
        y = []

        for match in goal_ratings:
            goal_rating = goal_ratings[match]
            expected_goal_rating = expected_goals_ratings[match]

            home_rating = goal_rating[0]
            away_rating = goal_rating[1]
            home_goal_diff = goal_rating[2]

            x_home_rating = expected_goal_rating[0]
            x_away_rating = expected_goal_rating[1]

            val = 0
            if home_goal_diff > 0:
                val = 0
            elif home_goal_diff == 0:
                val = 1
            else:
                val = 2
            x.append([home_rating - away_rating, x_home_rating - x_away_rating])
            y.append(val)

        (model, score) = random_search_cv_xgb(x, y)
        obj = V6(model, score, ratings_window_length)
        write_model("v6", obj)
        return obj
