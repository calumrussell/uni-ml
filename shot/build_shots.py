import json
import sys

from common import get_model_best_score

if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit(1)

    period = sys.argv[1]
    if period != "test" and period != "train":
        exit(1)

    best_model = get_model_best_score()
    matches = {}

    with open(f"data/{period}_shots.json") as f:
        shots = []
        for line in f.readlines():
            shot = json.loads(line)
            shots.append(shot)

        preds = best_model.predict(shots)
        for shot, pred in zip(shots, preds):
            result = shot['result']
            match_id = int(shot['match_id'])
            team_id = int(shot['team_id'])

            score_prob = pred[1]
            if match_id not in matches:
                matches[match_id] = {}

            if team_id not in matches[match_id]:
                matches[match_id][team_id] = 0

            matches[match_id][team_id] += score_prob

    with open(f"data/{period}_expected_goals.json", "w") as f:
        json.dump(matches, f)

