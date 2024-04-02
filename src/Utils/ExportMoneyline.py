import csv
import os
from datetime import datetime


def append_to_csv(p_team_name, p_win_prediction_confidence, p_current_odds, p_game_date,
                  p_file_name='out/moneyline_data.csv'):
    headers = ['Date', 'Game Date', 'Team Name', 'Win Prediction Confidence', 'Current Odds']

    if p_win_prediction_confidence is None:
        p_win_prediction_confidence = 0

    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if file exists and create it if it does not
    file_exists = os.path.isfile(p_file_name)

    with open(p_file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(headers)
        # Write the data with current date
        writer.writerow([current_date,
                         p_game_date,
                         p_team_name,
                         f"{p_win_prediction_confidence:.2f}".rjust(5),
                         str(p_current_odds).rjust(5)])
