import csv
import os
from datetime import datetime


def append_to_csv(home_team_odds, home_team_name, away_team_name, away_team_odds, predicted_winner,
                  win_prediction_confidence, over_under_point, over_under_prediction,
                  over_under_confidence, file_name='sports_data.csv'):
    headers = ['Date', 'Home Team Odds', 'Home Team Name', 'Away Team Name', 'Away Team Odds',
               'Predicted Winner', 'Win Prediction Confidence', 'Actual Winner', 'OverUnder Point',
               'OverUnder Prediction', 'OverUnder Confidence', 'Actual Combined Score']

    # Get current date
    current_date = datetime.now().strftime('%Y-%m-%d %h:%mi:%s')

    # Check if file exists and create it if it does not
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(headers)
        # Write the data with current date
        writer.writerow([current_date, home_team_odds, home_team_name, away_team_name, away_team_odds,
                         predicted_winner, win_prediction_confidence, "", over_under_point,
                         over_under_prediction, over_under_confidence, ""])


# Example usage
# append_to_csv(1.5, 'Team A', 'Team B', 2.0, 'Team A', 0.75, 'Team A', 50, 'Over', 0.6, 60)
