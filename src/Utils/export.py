import csv
import os
from datetime import datetime


def append_to_csv(home_team_odds, home_team_name, away_team_name, away_team_odds, predicted_winner,
                  win_prediction_confidence, over_under_point, over_under_prediction,
                  under_over_confidence, kelly_percentage, file_name='out/sports_data.csv'):
    headers = ['Date', 'Win Prediction Confidence', 'Kelly Percentage',
               'Home Team Odds', 'Away Team Odds',
               'OverUnder Prediction', 'OverUnder Point', 'OverUnder Confidence', 'Actual Combined Score',
               'Home Team Name', 'Away Team Name',
               'Predicted Winner', 'Actual Winner']

    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if file exists and create it if it does not
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(headers)
        # Write the data with current date
        writer.writerow([current_date, f"{win_prediction_confidence:.2f}".rjust(5), f"{kelly_percentage:.2f}".rjust(5),
                         str(home_team_odds).rjust(5), str(away_team_odds).rjust(5),
                         str(over_under_prediction).rjust(5), f"{over_under_point:.1f}".rjust(5),
                         f"{under_over_confidence:.2f}".rjust(5), "?",
                         home_team_name, away_team_name, predicted_winner, "?"])

# Example usage
# append_to_csv(1.5, 'Team A', 'Team B', 2.0, 'Team A', 0.75, 'Team A', 50, 'Over', 0.6, 60)
