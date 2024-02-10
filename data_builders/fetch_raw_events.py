import psycopg2
import os
import sys
import json

if __name__ == "__main__":
    """
    Used to fetch raw event-level data for use with shot prediction model. Hard-coded test
    and train sets.
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

    years_str = ",".join(years)

    res = {}
    conn = psycopg2.connect(os.getenv("DB_CONN"))
    with conn:
        with conn.cursor() as cur:

            cur.execute(f"select id from match where tournament_id=2 and year in ({years_str})")
            matches = [str(row[0]) for row in cur.fetchall()]
            matches_str = ",".join(matches)

            cur.execute(f"select id, data from match_data where id in ({matches_str})")
            for row in cur.fetchall():
                data = row[1]
                events = data['matchCentreData']['events']
                res[row[0]] = events

    with open(f"data/{period}_raw_events.json", 'w') as f:
        json.dump(res, f)
