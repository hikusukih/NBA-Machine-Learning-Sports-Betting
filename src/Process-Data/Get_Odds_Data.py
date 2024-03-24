import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
from sbrscrape import Scoreboard
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.tools import get_date

year = ["2023", "2024"]
season = ["2023-24"]

month = [10, 11, 12, 1, 2, 3, 4, 5, 6]
days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

begin_year_pointer = year[0]
end_year_pointer = year[0]
count = 0

sportsbook = 'fanduel'
df_data = []

con = sqlite3.connect("../../Data/odds.sqlite")

for iter_season in tqdm(season):
    teams_last_played = {}
    for iter_month in tqdm(month):
        if iter_month == 1:
            count += 1
            end_year_pointer = year[count]
        for iter_day in tqdm(days):
            if iter_month == 10 and iter_day < 24:
                continue
            if iter_month in [4, 6, 9, 11] and iter_day > 30:
                continue
            if iter_month == 2 and iter_day > 28:
                continue
            # skip future games
            if datetime.now() < datetime(year=int(end_year_pointer), month=iter_month, day=iter_day):
                continue
            print(f"{end_year_pointer}-{iter_month:02}-{iter_day:02}")
            sb = Scoreboard(date=f"{end_year_pointer}-{iter_month:02}-{iter_day:02}")
            if not hasattr(sb, "games"):
                continue
            for game in sb.games:
                if game['home_team'] not in teams_last_played:
                    teams_last_played[game['home_team']] = get_date(f"{iter_season}-{iter_month:02}{iter_day:02}")
                    home_games_rested = timedelta(days=7)  # start of season, big number
                else:
                    current_date = get_date(f"{iter_season}-{iter_month:02}{iter_day:02}")
                    home_games_rested = current_date - teams_last_played[game['home_team']]
                    teams_last_played[game['home_team']] = current_date
                    # todo update row

                if game['away_team'] not in teams_last_played:
                    teams_last_played[game['away_team']] = get_date(f"{iter_season}-{iter_month:02}{iter_day:02}")
                    away_games_rested = timedelta(days=7)  # start of season, big number
                else:
                    current_date = get_date(f"{iter_season}-{iter_month:02}{iter_day:02}")
                    away_games_rested = current_date - teams_last_played[game['away_team']]
                    teams_last_played[game['away_team']] = current_date

                try:
                    df_data.append({
                        'Unnamed: 0': 0,
                        'Date': f"{iter_season}-{iter_month:02}{iter_day:02}",
                        'Home': game['home_team'],
                        'Away': game['away_team'],
                        'OU': game['total'][sportsbook],
                        'Spread': game['away_spread'][sportsbook],
                        'ML_Home': game['home_ml'][sportsbook],
                        'ML_Away': game['away_ml'][sportsbook],
                        'Points': game['away_score'] + game['home_score'],
                        'Win_Margin': game['home_score'] - game['away_score'],
                        'Days_Rest_Home': home_games_rested.days,
                        'Days_Rest_Away': away_games_rested.days
                    })
                except KeyError:
                    print(f"No {sportsbook} odds data found for game: {game}")
            time.sleep(random.randint(1, 3))
    begin_year_pointer = year[count]

    df = pd.DataFrame(df_data, )
    df.to_sql(f"odds_{iter_season}", con, if_exists="replace")
con.close()
