import psycopg2
import psycopg2.extras
import os
import sys
import json

if __name__ == "__main__":
    """
    Used to fetch match result data for use with match prediction model. Hard-coded test
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
        with conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor) as cur:

            ## We only need to retrieve home row of team_stats_full for match outcome
            cur.execute(f"select id, home_id, start_date from match where tournament_id=2 and year in ({years_str})")
            matches = [(row["id"], row["home_id"], row["start_date"]) for row in cur.fetchall()]

            for match_id, home_id, start_date in matches:
                cur.execute(f"select goal_for, goal_against, is_win, is_draw, is_loss, team_id, opp_id, match_id from team_stats_full where match_id={match_id} and team_id={home_id}")
                res[match_id] = dict({**cur.fetchone(), **{"start_date": start_date}})

    with open(f"data/{period}_match_results.json", 'w') as f:
        json.dump(res, f)
