* Set up data for shots
    * Split into test and validation
    * Will use 2018-2022 as test, validate with 23
    * Raw event data will need to be base (but shouldn't be put in final? need to check compressed size)
    * Will need to version intermediate datasets
    * Model code also needs to be versioned, can't use source control

* Set up data for matches
    * Problem with this is that we are using xG as an input to matches


```
source venv/bin/activate
python data_builders/fetch_raw_events.py test
python data_builders/fetch_raw_events.py train
python data_builders/parse_shots.py test
python data_builders/parse_shots.py train
python data_builders/fetch_match_results.py test
python data_builders/fetch_match_results.py train
```
