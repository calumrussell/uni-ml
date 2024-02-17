from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import poisson
import numpy as np
from sklearn.preprocessing import LabelBinarizer

from common import get_train_set, PoissonRatingTrainer, get_train_set_expected_goals, make_brier_multi_scorer_with_lb, brier_multi, write_model, get_model_best_score

def random_search_cv_logistic(x, y):

    params = {
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    logistic_model = LogisticRegression(multi_class="multinomial")
    lb = LabelBinarizer()
    lb.fit_transform(y)

    random_search = RandomizedSearchCV(
            logistic_model,
            param_distributions=params,
            scoring=make_brier_multi_scorer_with_lb(lb=lb),
            n_jobs=5,
            cv=5,
    )
    random_search.fit(x,y)
    
    return (random_search.best_estimator_, random_search.best_score_)

class V2PoissonRatings:
    def __init__(self, goal_ratings, expected_goals_ratings):
        self.goal_ratings = goal_ratings
        self.expected_goals_ratings = expected_goals_ratings

    @staticmethod
    def train(window_length):
        goal_trainer = PoissonRatingTrainer(window_length)
        expected_goal_trainer = PoissonRatingTrainer(window_length)

        match_results = get_train_set()
        expected_goals = get_train_set_expected_goals()
        for match in match_results:
            match_id = match["match_id"]
            home_id = match["team_id"]
            away_id = match["opp_id"]
            
            match_expected_goals = expected_goals[str(match_id)]

            home_xg = match_expected_goals.get(str(home_id))
            away_xg = match_expected_goals.get(str(away_id))
            if not home_xg:
                home_xg = 0
            if not away_xg:
                away_xg = 0

            goal_trainer.update(home_id, away_id, match["goal_for"], match["goal_against"], match["start_date"], match["match_id"])
            expected_goal_trainer.update(home_id, away_id, home_xg, away_xg, match["start_date"], match["match_id"])

        goal_ratings = goal_trainer.get_ratings()
        xg_ratings = expected_goal_trainer.get_ratings()
        return V2PoissonRatings(goal_ratings, xg_ratings)

class V2PoissonToLogisticProbability:
    def __init__(self, model, score):
        self.model = model
        self.score = score

    def predict(self, home_rating, away_rating):
        rating_diff = home_rating - away_rating
        pred = self.model.predict_proba([[rating_diff]])
        return {i: j for i, j in zip(self.model.classes_, pred[0])}

    @staticmethod
    def train(ratings_window_length):
        ratings_model = V2PoissonRatings.train(ratings_window_length)
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

            result = "loss"
            if home_goal_diff > 0:
                result = "win"
            elif home_goal_diff == 0:
                result = "draw"
            x.append([home_rating - away_rating, x_home_rating - x_away_rating])
            y.append(result)

        (model, score) = random_search_cv_logistic(x, y)
        obj = V2PoissonToLogisticProbability(model, score)
        write_model("v1_logistic", obj)
        return obj

if __name__ == "__main__":
    v2 = V2PoissonToLogisticProbability.train(50)
    print(v2.score)
