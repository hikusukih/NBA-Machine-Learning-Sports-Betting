import os
import random
import sqlite3
import sys
import time
from calendar import monthrange
from datetime import date, datetime, timedelta

from tqdm import tqdm

# Put Utils on the classpath
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.tools import get_json_data, to_data_frame

"""
This script is responsible for **gathering NBA team statistics** from the NBA stats API
and storing the data in an SQLite database.

Attributes:
    url (str): The URL template for scraping NBA team statistics.
    year (list): List of years to scrape.
    season (list): List of NBA seasons (formatted as strings) to scrape.
    month (list): List of months (October through June) to iterate over in each NBA season.
    days (list): List of days (1-31) to iterate through for each month.
    begin_year_pointer (int): Tracks the starting year in the current iteration.
    end_year_pointer (int): Tracks the ending year in the current iteration.
    count (int): Counter for advancing through years in the iteration.

Database:
    - The scraped data is stored in an SQLite database, with table names in the format `teams_{season}-{month}-{day}`.

Flow:
    1. **Initialize**: Set up the database connection and the iteration variables.
    2. **Gather Data**: Loop over seasons, months, and days to collect the raw data.
    3. **Store Data**: Insert the collected data into an SQLite database.
    4. **Close**: Ensure the database connection is properly closed after the data collection is complete.

Note: This script uses the `tqdm` library to display progress bars for the season, month, and day iterations.
"""


url = 'https://stats.nba.com/stats/' \
      'leaguedashteamstats?Conference=&' \
      'DateFrom=10%2F01%2F{2}&DateTo={0}%2F{1}%2F{3}' \
      '&Division=&GameScope=&GameSegment=&LastNGames=0&' \
      'LeagueID=00&Location=&MeasureType=Base&Month=0&' \
      'OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&' \
      'PerMode=PerGame&Period=0&PlayerExperience=&' \
      'PlayerPosition=&PlusMinus=N&Rank=N&' \
      'Season={4}' \
      '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
      'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
# ALL years. Don't do this more than once, you saved it already!
# year = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
# season = ["2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
#           "2017-18", "2018-19", "2019-20", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]

year = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
season = ["2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
          "2017-18", "2018-19", "2019-20", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]

# year = [2023, 2024] # Update with the current/recent season data
# season = ["2023-24"] # update with the current/recent season data

# October through June
month = [10, 11, 12, 1, 2, 3, 4, 5, 6]
days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31]

begin_year_pointer = year[0]
end_year_pointer = year[0]
count = 0

con = sqlite3.connect("../../Data/teams.sqlite")

# Iterate through all seasons/years/months/days. "pb_" stands for "progress bar"
with tqdm(total=len(season)) as pb_season:
    for iter_season in season:
        pb_season.set_description(f"Season: {iter_season}")
        with tqdm(total=len(month)) as pb_month:
            for iter_month in month:
                pb_month.set_description(f"Months")
                if iter_month == 1:
                    count += 1
                    end_year_pointer = year[count]
                with tqdm(total=len(days)) as pb_day:
                    for iter_day in days:
                        pb_day.set_description(f"{year[count]}-{iter_month:02}-{iter_day:02}")
                        # No games before October 24 (use 15 just in case)
                        if iter_month == 10 and iter_day < 15:
                            pb_day.update(1)
                            continue
                        # Check if the day is valid for the month and year
                        if iter_day > monthrange(end_year_pointer, iter_month)[1]:
                            pb_day.update(1)
                            continue
                        # Don't get game data about "today"
                        if end_year_pointer == datetime.now().year:
                            if iter_month == datetime.now().month and iter_day > datetime.now().day:
                                pb_day.update(1)
                                continue
                            if iter_month > datetime.now().month:
                                pb_day.update(1)
                                continue

                        url_formatted = url.format(iter_month, iter_day, begin_year_pointer, end_year_pointer, iter_season)
                        general_data = get_json_data(url_formatted)
                        general_df = to_data_frame(general_data)

                        try:
                            real_date = date(year=end_year_pointer, month=iter_month, day=iter_day) + timedelta(days=1)
                        except ValueError:
                            pb_day.update(1)
                            continue

                        general_df['Date'] = str(real_date)

                        x = str(real_date).split('-')
                        general_df.to_sql(f"teams_{iter_season}-{str(int(x[1]))}-{str(int(x[2]))}", con=con, if_exists="replace")

                        # Rest a while, randomize the requests
                        time.sleep(random.randint(2, 4))
                        pb_day.update(1)
                pb_month.update(1)
        begin_year_pointer = year[count]
        pb_season.update(1)
con.close()
