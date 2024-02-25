data/           - data used as input to models, inc intermediate data

data_builders/  - scripts used to create base datasets

match/          - match prediction model

shot/           - shot prediction model

build.sh - used to fetch data and write to file, can't be run without DB source

train.sh - used to train both models, write expected goals data to file using best shot model
before starting match prediction.

test.sh - reports test performance of all models on file
