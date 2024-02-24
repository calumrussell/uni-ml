from scipy.stats import poisson
import numpy as np

from common import PoissonRatings, brier_multi, write_model

class V4:
    """
    Uses Poisson pmf to calculate goal probabilities from goal ratings (which
    are just Poisson variables).
    """
    def __init__(self, score, ratings_window_length):
        self.score = score
        self.ratings_window_length = ratings_window_length

    def predict(self, home_rating, away_rating):
        probs = V4.calc_probability(home_rating, away_rating)
        return {"win": probs[0], "draw": probs[1], "loss": probs[2]}

    @staticmethod
    def calc_probability(home_rating, away_rating):
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
        return (win_prob, draw_prob, loss_prob)

    def test_score(self):
        ratings_model = PoissonRatings.test(self.ratings_window_length)
        ratings = ratings_model.goal_ratings

        match_probs = []
        outcomes = []

        for match in ratings:
            rating = ratings[match]

            home_rating = rating[0]
            away_rating = rating[1]
            home_goal_diff = rating[2]

            (win_prob, draw_prob, loss_prob) = V4.calc_probability(
                    home_rating, away_rating)

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
        #We flip sign of score as we want to maximise
        return brier_multi(outcomes, match_probs) * -1

    @staticmethod
    def train(ratings_window_length):
        ratings_model = PoissonRatings.train(ratings_window_length)
        ratings = ratings_model.goal_ratings

        match_probs = []
        outcomes = []

        for match in ratings:
            rating = ratings[match]

            home_rating = rating[0]
            away_rating = rating[1]
            home_goal_diff = rating[2]

            (win_prob, draw_prob, loss_prob) = V4.calc_probability(
                    home_rating, away_rating)

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
        #We flip sign of score as we want to maximise
        score = brier_multi(outcomes, match_probs) * -1
        obj = V4(score, ratings_window_length)
        write_model("v4", obj)
        return obj

