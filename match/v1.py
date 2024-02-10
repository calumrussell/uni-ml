import os

from common import get_train_set

from sklearn.linear_model import PoissonRegressor

class V1Trainer:

    def __init__(self):
        self.matches = {}
        self.ratings = []
        self.window_length = 10
        ## Dataset shouldn't have double records but this is included to make sure that the same match
        ## isn't counted twice (which can happen if you have a row for the home and away team)
        self.calculated = {}
        return

    def _optimize(self):
        teams = self.matches.keys()
        for team in teams:
            team_matches = self.matches[team]
            if len(team_matches) > self.window_length:
                window = team_matches[-self.window_length:]
            else:
                window = team_matches

            ## The date that the rating is before the last result
            last_date = window[-1][2]
            if hash(str(team) + str(last_date)) not in self.calculated:
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
                self.ratings.append([team, off_rating, def_rating, last_date])

    def update(self, home_team, away_team, home_goals, away_goals, date):
        if home_team not in self.matches:
            self.matches[home_team] = []
        if away_team not in self.matches:
            self.matches[away_team] = []

        self.matches[home_team].append([home_goals, away_goals, date])
        self.matches[away_team].append([away_goals, home_goals, date])
        self._optimize()
        return

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
        Train should return a copy of self with the actual completed model. So train is an object
        creator.
        """
        trainer = V1Trainer()

        results = get_train_set()
        for match in results:
            home_id = match["team_id"]
            away_id = match["opp_id"]

            trainer.update(home_id, away_id, match["goal_for"], match["goal_against"], match["start_date"])
        print(trainer.ratings)


if __name__ == "__main__":

    v1 = V1.train()
