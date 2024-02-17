from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import poisson
import numpy as np
from sklearn.preprocessing import LabelBinarizer

from common import get_train_set, PoissonRatingTrainer, make_brier_multi_scorer_with_lb, brier_multi

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

class V1PoissonRatings:
    """
    Intermediate modelling stage that builds a set of ratings for each team over a window period.

    We need an intermediate stage because we deploy two methods to map a pair of team ratings to
    a match outcome.

    The rating model used is composed of an independent Poisson for the offensive and defensive
    strength of each team. The team's goal scored/goals conceded over the last N matches is used
    as the dependent variable for the offensive and defensive model respectively. The independent
    variable is just a dummy 1 value representing team identity (it is possible, of course, to model
    all teams simultaneously from a certain date but it is easier to do the regressions seperately
    in practice to ensure that windows don't overlap, you aren't using "future" data, etc.).
    """
    def __init__(self, ratings):
        self.ratings = ratings

    @staticmethod
    def train(window_length):
        """
        Builds ratings over the training set. This is unintuitive as there can be no "training" set
        with this kind of model as we have no idea what actual strength is. However, we may use
        `V1PoissonToLogisticProbability` later which maps ratings to probability and this does have
        real notions of "fit" so we want to hold out the test set until later.
        """
        trainer = PoissonRatingTrainer(window_length)

        results = get_train_set()
        for match in results:
            home_id = match["team_id"]
            away_id = match["opp_id"]

            trainer.update(home_id, away_id, match["goal_for"], match["goal_against"], match["start_date"], match["match_id"])

        ratings = trainer.get_ratings()
        return V1PoissonRatings(ratings)

class V1PoissonToLogisticProbability:
    """
    Goes from team rating to match probability.

    Can use train-test split here.
    """

    @staticmethod
    def train(ratings_window_length):
        ratings_model = V1PoissonRatings.train(ratings_window_length)
        ratings = ratings_model.ratings

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
        print(score)
        return

    def __init__(self):
        return

class V1PoissonToPoissonProbability:
    """
    Goes from team rating to match probability using Poisson pmf.
    """

    @staticmethod
    def train(ratings_window_length):
        ratings_model = V1PoissonRatings.train(ratings_window_length)
        ratings = ratings_model.ratings

        match_probs = []
        outcomes = []

        for match in ratings:
            rating = ratings[match]

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


if __name__ == "__main__":

    V1PoissonToPoissonProbability.train(50)
    V1PoissonToLogisticProbability.train(50)
