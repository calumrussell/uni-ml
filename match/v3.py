import numpy as np
from scipy.special import binom, factorial
from scipy.optimize import minimize

from common import brier_multi, PoissonRatings, write_model

def calculate_merged_prob(goal_rating, expected_goal_rating, corr_goal, corr_expected, weight):
    home_rating = goal_rating[0]
    away_rating = goal_rating[1]

    x_home_rating = expected_goal_rating[0]
    x_away_rating = expected_goal_rating[1]

    (win_prob, draw_prob, loss_prob) = V3.calc_probability(
                home_rating, away_rating, corr_goal)

    (x_win_prob, x_draw_prob, x_loss_prob) = V3.calc_probability(
            x_home_rating, x_away_rating, corr_expected)

    merged_win_prob = (win_prob * (1-weight)) + (x_win_prob * weight)
    merged_draw_prob = (draw_prob * (1-weight)) + (x_draw_prob * weight)
    merged_loss_prob = (loss_prob * (1-weight)) + (x_loss_prob * weight)
    return (merged_win_prob, merged_draw_prob, merged_loss_prob)

def bipoiss_pmf(x, y, a, b, c, acc=50):
    if x < 0 or y < 0:
        raise ValueError

    first = np.exp(-(a+b+c)) * ((a**x)/factorial(x) * (b**y)/factorial(y)) 
    vals = []
    vals.append(binom(x, 0) * binom(y, 0) * factorial(0) * (c/(a*b)) ** 0)
    for k in range(1, min(x,y)):
        vals.append(binom(x, k) * binom(y, k) * factorial(k) * (c/(a*b)) ** k)
    return first * sum(vals)

def _loss_bipoisson(par, matches, goal_ratings, expected_goal_ratings):
    if par[0] < 0 or par[0] >= 1:
        return np.inf

    if par[1] < 0 or par[1] >= 1:
        return np.inf

    if par[2] < 0.1 or par[2] >=0.9:
        return np.inf

    match_probs = []
    outcomes = []
    for match in matches:
        goal_rating = goal_ratings[match]
        expected_goal_rating = expected_goal_ratings[match]

        home_goal_diff = goal_rating[2]
        (win_prob, draw_prob, loss_prob) = calculate_merged_prob(
                goal_rating, expected_goal_rating, par[0], par[1], par[2])
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
    score = brier_multi(outcomes, match_probs)
    ##Higher brier is lower so this will be minimized normally
    return score

class V3:
    def __init__(self, score, goal_corr, expected_goal_corr, weight, window_length):
        self.score = score
        self.goal_corr = goal_corr
        self.expected_goal_corr = expected_goal_corr
        self.weight = weight
        self.window_length = window_length

    def test_score(self):
        ratings_model = PoissonRatings.test(self.window_length)
        goal_ratings = ratings_model.goal_ratings
        expected_goal_ratings = ratings_model.expected_goals_ratings

        match_probs = []
        outcomes = []

        for match in goal_ratings:
            goal_rating = goal_ratings[match]
            expected_goal_rating = expected_goal_ratings[match]

            home_goal_diff = goal_rating[2]
            (win_prob, draw_prob, loss_prob) = calculate_merged_prob(
                    goal_rating, expected_goal_rating, self.goal_corr, self.expected_goal_corr, self.weight)
     
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
    def calc_probability(home_rating, away_rating, corr):
        probs = []
        for i in range(0, 10):
            for j in range(0, 10):
                probs.append(bipoiss_pmf(i, j, home_rating, away_rating, corr))

        split = [probs[i:i+10] for i in range(0,len(probs),10)]
        draw_prob = (np.sum(np.diag(split)))
        win_prob = (np.sum(np.tril(split, -1)))
        loss_prob = (np.sum(np.triu(split, 1)))
        return (win_prob, draw_prob, loss_prob)

    @staticmethod
    def train(ratings_window_length):
        ratings_model = PoissonRatings.train(ratings_window_length)
        goal_ratings = ratings_model.goal_ratings
        expected_goal_ratings = ratings_model.expected_goals_ratings


        # Average all params
        tmp_param = []

        for param in np.random.uniform(low=0.1, high=1.0, size=(1, 3)):
            matches_subset = np.random.choice(list(goal_ratings.keys()), size=200)
            res = minimize(
                fun=_loss_bipoisson,
                method='Nelder-Mead',
                x0=param,
                args=(matches_subset, goal_ratings, expected_goal_ratings),
            )
            tmp_param.append(res.x)
        param = np.mean(tmp_param, axis=0)

        match_probs = []
        outcomes = []
        for match in goal_ratings:
            goal_rating = goal_ratings[match]
            expected_goal_rating = expected_goal_ratings[match]

            home_goal_diff = goal_rating[2]
            (win_prob, draw_prob, loss_prob) = calculate_merged_prob(
                    goal_rating, expected_goal_rating, param[0], param[1], param[2])
 
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
        score = brier_multi(outcomes, match_probs) * -1
        obj = V3(score, param[0], param[1], param[2], ratings_window_length)
        write_model("v3", obj)
        return obj

if __name__ == "__main__":
    v3 = V3.train(30)
    print(v3.score)
