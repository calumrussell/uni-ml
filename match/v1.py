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

        match_predictions = trainer.get_predictions()

        actuals = []
        preds = []
        for match in match_predictions:
            actuals.extend(match_predictions[match][1])
            preds.extend(match_predictions[match][0])

        print(brier_score_loss(actuals, preds))
        return

if __name__ == "__main__":

    v1 = V1.train()
