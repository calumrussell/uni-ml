import json

def get_train_set():
    with open('data/train_match_results.json') as f:
        match_results = json.load(f)

    tmp = [match_results[match] for match in match_results]
    return sorted(tmp, key=lambda x: x['start_date'])
