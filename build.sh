python data_builders/parse_shots.py test
python data_builders/parse_shots.py train
python data_builders/fetch_match_results.py test
python data_builders/fetch_match_results.py train
python data_builders/fetch_match_odds.py test
python data_builders/fetch_match_odds.py train
zstd -fr ./data/*.json
