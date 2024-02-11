import os
from scipy.stats import poisson
import numpy as np
from sklearn.metrics import brier_score_loss

from common import get_train_set, PoissonTrainer

class V1:
    """
    Basic model used for testing:
        * Team
    """
    def __init__(self, model, score, x, y):
        self.model = model
        self.score = score
        self.x = x
        self.y = y

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
        ratings = trainer.ratings

        grouped_ratings = []
        for match in ratings:
            if len(ratings[match]) != 2:
                continue

            home_idx = 0 if ratings[match][0][5] == 1 else 0
            away_idx = 0 if home_idx == 1 else 1

            home_off_rating = ratings[match][home_idx][1]
            home_def_rating = ratings[match][home_idx][2]
            away_off_rating = ratings[match][away_idx][1]
            away_def_rating = ratings[match][away_idx][2]

            home_goal_diff = ratings[match][home_idx][4]
            result = 'loss'
            if home_goal_diff > 0:
                result = 'win'
            elif home_goal_diff == 0:
                result = 'draw'

            grouped_ratings.append([home_off_rating, home_def_rating, away_off_rating, away_def_rating, result])

        actuals = []
        pred = []
        for rating in grouped_ratings:
            ## Convert the Poisson variables into a goal prediction

            ## Multiply home off rating by away def rating to get expected goals for home
            ## Multiply away off rating by home def rating to get expected goals for away
            home_exp = rating[0] * rating[3]
            away_exp = rating[2] * rating[1]

            probs = []
            for i in range(0, 10):
                for j in range(0, 10):
                    home_prob = poisson.pmf(i, home_exp)
                    away_prob = poisson.pmf(j, away_exp)
                    probs.append(home_prob * away_prob)

            split = [probs[i:i+10] for i in range(0,len(probs),10)]

            draw_prob = (np.sum(np.diag(split)))
            win_prob = (np.sum(np.tril(split, -1)))
            loss_prob = (np.sum(np.triu(split, 1)))

            outcome = rating[4]
            win = 1 if outcome == "win" else 0
            draw = 1 if outcome == "draw" else 0
            loss = 1 if outcome == "loss" else 0

            pred.extend([win_prob, draw_prob, loss_prob])
            actuals.extend([win, draw, loss])
        print(brier_score_loss(actuals, pred))



if __name__ == "__main__":

    v1 = V1.train()
