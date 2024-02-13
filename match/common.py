import json

from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson
import numpy as np

class PoissonTrainer:

    def __init__(self):
        self.matches = {}
        ## Indexed by match_id
        self.ratings = {}
        self.window_length = 50
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

    def get_predictions(self):
        match_predictions = {}

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

            win = 0
            draw = 0
            loss = 0
            if home_goal_diff > 0:
                win = 1
            elif home_goal_diff == 0:
                draw = 1
            else:
                loss = 1

            match_predictions[match] = [(win_prob, draw_prob, loss_prob), (win, draw, loss)]
        return match_predictions

def get_train_set():
    with open('data/train_match_results.json') as f:
        match_results = json.load(f)

    tmp = [match_results[match] for match in match_results]
    return sorted(tmp, key=lambda x: x['start_date'])
