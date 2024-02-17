import sys
import json
import requests

def get_year_odds(year):
    first_year_str = year[-2:]
    second_year_str = str(int(first_year_str) + 1)
    url = f"https://www.football-data.co.uk/mmz4281/{first_year_str}{second_year_str}/E0.csv"
    return requests.get(url).content.decode('utf-8')

def parse_csv(csv):
    ##I believe that position of the cols move about over the years so rather than hardcode positions
    ##will just look for certain cols for safety
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'PSH', 'PSD', 'PSA']
    csv_split = csv.split("\r\n")
    headers = csv_split[0].split(",")

    data_pos = []
    ##Cols are definitely ordered so we can use this shortcut
    cols_pos = 0
    for i, header in enumerate(headers):
        if header == cols[cols_pos]:
            data_pos.append(i)
            if cols_pos >= len(cols) - 1:
                break
            else:
                cols_pos +=1

    data = []
    for raw_row in csv_split[1:-1]:
        row = raw_row.split(",")
        tmp = []
        for pos in data_pos:
            tmp.append(row[pos])
        data.append(tmp)
    return data

def parse_odds(raw_odds):
    """
    Odds from bookmakers do not represent true probability because they include the bookmaker's profit
    margin. To calculate "true probability" from European odds we:
        * convert from odds to probability
        * add up the total probability of all outcomes (will be more than one)
        * divide the probability of each outcome by the total probability
    We also use odds from the bookmaker Pinnacle which is known to have a low "profit margin" and prices
    outcomes based on the flow they get from customers. Some bookmakers may choose to price an event
    inaccurately based on imbalances in their book, this makes calculations of true odds from their prices
    difficult. There are ways to correct this (for example, correcting for things like longshot bias) but
    it is simpler to use a model without parameters that is roughly correct.
    """
    true_odds = []
    for odds in raw_odds:
        tmp = {}

        tmp['Date'] = odds[0]
        tmp['Home'] = odds[1]
        tmp['Away'] = odds[2]
        tmp['HomeScore'] = odds[3]
        tmp['AwayScore'] = odds[4]

        home_prob = 1/float(odds[5])
        draw_prob = 1/float(odds[6])
        away_prob = 1/float(odds[7])

        total = home_prob + draw_prob + away_prob

        true_home = home_prob/total
        true_draw = draw_prob/total
        true_away = away_prob/total

        tmp['HomeProb'] = true_home
        tmp['DrawProb'] = true_draw
        tmp['AwayProb'] = true_away
        true_odds.append(tmp)
    return true_odds

if __name__ == "__main__":
    """
    Fetches match odds for benchmarking match prediction model. Hard-coded test and
    train sets.
    """
    if len(sys.argv) != 2:
        exit(1)

    period = sys.argv[1]
    if period != "test" and period != "train":
        exit(1)

    if period == "test":
        years = ['2022', '2023']
    else:
        years = ['2017', '2018', '2019', '2020', '2021']

    odds = []
    for year in years:
        odds_csv = get_year_odds(year)
        parsed_odds = parse_csv(odds_csv)
        true_odds = parse_odds(parsed_odds)
        odds.extend(true_odds)


    with open(f"data/{period}_match_odds.json", 'w') as f:
        for odd in odds:
            f.write(json.dumps(odd) + "\r\n")
        
