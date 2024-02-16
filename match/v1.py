from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import poisson
import numpy as np
from sklearn.preprocessing import LabelBinarizer, label_binarize

from common import get_train_set, PoissonTrainer

def brier_multi(targets, probs):
    return np.mean(np.sum((np.array(probs) - np.array(targets))**2, axis=1))

def brier_multi_lb_wrapper(targets, probs, lb):
    reversed = lb.transform(targets)
    return brier_multi(reversed, probs)

def make_brier_multi_scorer(lb):
    """
    When you have greater_is_better is multiplies the score by -1 and maximises.
    """
    return make_scorer(score_func=brier_multi_lb_wrapper, greater_is_better=False, response_method='predict_proba', lb=lb)

def random_search_cv_logistic(x, y):

    params = {
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    logistic_model = LogisticRegression(multi_class="multinomial", max_iter=1000)
    lb = LabelBinarizer()
    lb.fit_transform(y)

    random_search = RandomizedSearchCV(
            logistic_model,
            param_distributions=params,
            scoring=make_brier_multi_scorer(lb=lb),
            n_jobs=5,
            cv=10,
            verbose=3,
            random_state=100
    )
    random_search.fit(x,y)
    
    return (random_search.best_estimator_, random_search.best_score_)

class V1_Name:
    """
    Goes from team rating to match probability.

    Can use train-test split here.
    """

    @staticmethod
    def train(v1_ratings):
        x = []
        y = []

        for match in v1_ratings.ratings:
            rating = v1_ratings.ratings[match]

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
        print(score)
        return

    def __init__(self):
        return

class V1_AnotherName:
    """
    Goes from team rating to match probability using Poisson pmf.
    """

    @staticmethod
    def train(v1_ratings):
        match_probs = []
        outcomes = []

        for match in v1_ratings.ratings:
            rating = v1_ratings.ratings[match]

            home_rating = rating[0]
            away_rating = rating[1]
            home_goal_diff = rating[2]

            probs = []
            for i in range(0, 10):
                for j in range(0, 10):
                    home_prob = poisson.pmf(i, home_rating)
                    away_prob = poisson.pmf(j, away_rating)
                    probs.append(home_prob * away_prob)

            split = [probs[i:i+10] for i in range(0,len(probs),10)]

            draw_prob = (np.sum(np.diag(split)))
            win_prob = (np.sum(np.tril(split, -1)))
            loss_prob = (np.sum(np.triu(split, 1)))

            win = 0
            draw = 0
            loss = 0

            if home_goal_diff > 0:
                win = 1
            elif home_goal_diff == 0:
                draw = 1
            else:
                loss = 1

            match_probs.append((win_prob, draw_prob, loss_prob))
            outcomes.append([win, draw, loss])
        print(brier_multi(outcomes, match_probs))
        return

    def __init__(self):
        return

class V1Ratings:
    """
    Basic model used for testing:
        * Team
    """
    def __init__(self, ratings):
        self.ratings = ratings

    @staticmethod
    def train():
        """
        Training is more complex because we are building X Poisson regressions over Y*2*2 matches where
        X is the number of teams and Y is the number of matches (one for each team, one for off/def).

        To build a prediction for a match we need to get the off/def ratings for each team using all matches
        up until that point.
        """
        trainer = PoissonTrainer()

        results = get_train_set()
        for match in results:
            home_id = match["team_id"]
            away_id = match["opp_id"]

            trainer.update(home_id, away_id, match["goal_for"], match["goal_against"], match["start_date"], match["match_id"])

        ratings = trainer.get_ratings()
        return V1Ratings(ratings)

if __name__ == "__main__":

    v1_ratings = V1Ratings.train()

    V1_Name.train(v1_ratings)
    V1_AnotherName.train(v1_ratings)
