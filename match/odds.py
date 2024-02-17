import json

import numpy as np

from common import brier_multi

def parse_row(row):
    win = 0
    draw = 0
    loss = 0
    if row['HomeScore'] == row['AwayScore']:
        draw = 1
    elif row['HomeScore'] > row['AwayScore']:
        win = 1
    else:
        loss = 1
    return ([win, draw, loss], [row['HomeProb'], row['DrawProb'], row['AwayProb']])

class Odds:

    @staticmethod
    def test():
        actuals = []
        preds = []
        with open('data/test_match_odds.json') as f:
            for line in f.readlines():
                row = parse_row(json.loads(line))
                actuals.append(row[0])
                preds.append(row[1])
        print(brier_multi(actuals, preds))
        return

    @staticmethod
    def bootstrap_train(n, window):
        actuals = []
        preds = []

        rows = []
        with open('data/train_match_odds.json') as f:
            for line in f.readlines():
                rows.append(line)

        for i in range(n):
            matches = np.random.choice(rows, window, replace=False)
            tmp_actuals = []
            tmp_preds = []
            for match in matches:
                row = parse_row(json.loads(match))
                tmp_actuals.append(row[0])
                tmp_preds.append(row[1])
            actuals.append(tmp_actuals)
            preds.append(tmp_preds)

        results = []
        for a, p in zip(actuals, preds):
            results.append(brier_multi(a, p))
        print(np.mean(results), np.median(results), np.std(results))
        return

    @staticmethod
    def train():
        actuals = []
        preds = []
        with open('data/train_match_odds.json') as f:
            for line in f.readlines():
                row = parse_row(json.loads(line))
                actuals.append(row[0])
                preds.append(row[1])
        print(brier_multi(actuals, preds))
        return

if __name__ == "__main__":
    Odds.train()
    Odds.test()
    Odds.bootstrap_train(50, 100)
