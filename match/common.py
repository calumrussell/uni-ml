import json
import time
import pickle
import os

from sklearn.metrics import make_scorer
from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson
import numpy as np

class PoissonRatingTrainer:
    """
    Offensive and defensive ratings for each team for the next match.

    We build X*Y*2 Poisson regressions where X is the number of teams and Y is the number of 
    matches with regressions for offensive and defensive strength.

    Window_length is a hyperparameter. Represents the maximum number of matches used in regressions
    and below this level all matches are used.
    """
    def __init__(self, window_length):
        self.matches = {}
        ## Indexed by match_id
        self.ratings = {}
        self.window_length = window_length
        ## Dataset shouldn't have double records but this is included to make sure that the same match
        ## isn't counted twice (which can happen if you have a row for the home and away team)
        self.calculated = {}
        return

    def _optimize(self):
        teams = self.matches.keys()
        for team in teams:
            team_matches = self.matches[team]
            ## If we only have one match, we can't use this to predict
            if len(team_matches) < 2:
                continue
            
            last_match_id = team_matches[-1][3]
            last_goals_for = team_matches[-1][0]
            last_goals_against = team_matches[-1][1]
            last_goal_diff = last_goals_for - last_goals_against
            last_is_home = team_matches[-1][4]
            if len(team_matches) > self.window_length:
                window = team_matches[-self.window_length:-1]
            else:
                window = team_matches[:-1]

            team_match_hash = hash(str(team) + str(last_match_id))
            if team_match_hash not in self.calculated:
                goals_for = [i[0] for i in window]
                goals_against = [i[1] for i in window]
                ## Each regression is composed of matches for a single team, so we are just aiming
                ## to recover the X value for a single team at a time
                team_input = [[1] for i in window]

                off_model = PoissonRegressor(fit_intercept=False, alpha=0)
                off_model.fit(team_input,goals_for)

                def_model = PoissonRegressor(fit_intercept=False, alpha=0)
                def_model.fit(team_input,goals_against)

                # We do this to recover the Poisson variable itself
                off_rating = off_model.predict([[1]])[0]
                def_rating = def_model.predict([[1]])[0]
                if last_match_id not in self.ratings:
                    self.ratings[last_match_id] = []
                self.ratings[last_match_id].append([team, off_rating, def_rating, last_match_id, last_goal_diff, last_is_home])
                self.calculated[team_match_hash] = 1

    def update(self, home_team, away_team, home_goals, away_goals, date, match_id):
        if home_team not in self.matches:
            self.matches[home_team] = []
        if away_team not in self.matches:
            self.matches[away_team] = []

        self.matches[home_team].append([home_goals, away_goals, date, match_id, 1])
        self.matches[away_team].append([away_goals, home_goals, date, match_id, 0])
        self._optimize()
        return

    def get_ratings(self):
        match_ratings = {}

        for match in self.ratings:
            if len(self.ratings[match]) != 2:
                continue

            home_idx = 0 if self.ratings[match][0][5] == 1 else 0
            away_idx = 0 if home_idx == 1 else 1

            home_off_rating = self.ratings[match][home_idx][1]
            home_def_rating = self.ratings[match][home_idx][2]
            away_off_rating = self.ratings[match][away_idx][1]
            away_def_rating = self.ratings[match][away_idx][2]

            home_goal_diff = self.ratings[match][home_idx][4]

            ## Multiply home off rating by away def rating to get expected goals for home
            ## Multiply away off rating by home def rating to get expected goals for away
            home_exp = home_off_rating * away_def_rating
            away_exp = away_off_rating * home_def_rating

            match_ratings[match] = (home_exp, away_exp, home_goal_diff)
        return match_ratings

class PoissonRatings:
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
    def __init__(self, goal_ratings, expected_goals_ratings, window_length):
        self.goal_ratings = goal_ratings
        self.expected_goals_ratings = expected_goals_ratings
        self.window_length = window_length

    @staticmethod
    def test(window_length):
        goal_trainer = PoissonRatingTrainer(window_length)
        expected_goal_trainer = PoissonRatingTrainer(window_length)

        results = get_test_set()
        expected_goals = get_test_set_expected_goals()
        for match in results:
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
        return PoissonRatings(goal_ratings, xg_ratings, window_length)

    @staticmethod
    def train(window_length):
        """
        Builds ratings over the training set. This is unintuitive as there can be no "training" set
        with this kind of model as we have no idea what actual strength is. However, we may use
        `V1PoissonToLogisticProbability` later which maps ratings to probability and this does have
        real notions of "fit" so we want to hold out the test set until later.
        """
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
        return PoissonRatings(goal_ratings, xg_ratings, window_length)

def get_test_set():
    with open('data/test_match_results.json') as f:
        match_results = json.load(f)

    tmp = [match_results[match] for match in match_results]
    return sorted(tmp, key=lambda x: x['start_date'])

def get_train_set():
    with open('data/train_match_results.json') as f:
        match_results = json.load(f)

    tmp = [match_results[match] for match in match_results]
    return sorted(tmp, key=lambda x: x['start_date'])

def get_train_set_expected_goals():
    with open('data/train_expected_goals.json') as f:
        expected_goals = json.load(f)
    return expected_goals

def get_test_set_expected_goals():
    with open('data/test_expected_goals.json') as f:
        expected_goals = json.load(f)
    return expected_goals

def brier_multi(targets, probs):
    return np.mean(np.sum((np.array(probs) - np.array(targets))**2, axis=1))

def brier_multi_lb_wrapper(targets, probs, lb):
    reversed = lb.transform(targets)
    return brier_multi(reversed, probs)

def make_brier_multi_scorer_with_lb(lb):
    """
    When you have greater_is_better is multiplies the score by -1 and maximises.
    """
    return make_scorer(score_func=brier_multi_lb_wrapper, greater_is_better=False, response_method='predict_proba', lb=lb)

def write_model(model_version, model_obj):
    epoch = round(time.time())
    
    with open(f"match/models/{model_version}_{epoch}.pkl", 'wb') as f:
        pickle.dump(model_obj, f)
    return

def get_model_best_score(model_version):
    last_score = -np.inf
    best_model = None
    models = os.listdir('match/models')
    for model_path in models:
        path, extension = model_path.split(".")
        version, model_type, epoch = path.split("_")
        if version == model_version:
            with open(f"match/models/{model_path}", "rb") as f:
                tmp_model = pickle.load(f)
                if tmp_model.score > last_score:
                    best_model = tmp_model
    return best_model

