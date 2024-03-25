import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc
from src.Utils import export as to_csv


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/XGBoost_67.7%_ML-4.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_54.2%_UO-9.json')
int_max_team_name_length = 22

def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    ml_predictions_array = []
    team_game_rec = {'home_team_odds': None, 'home_team_name': None,
                     'away_team_name': None, 'away_team_odds': None,
                     'predicted_winner': None, 'win_prediction_confidence': None,
                     'over_under_point': None,
                     'over_under_prediction': None, 'over_under_confidence': None}

    csv_output_array = [team_game_rec] * len(data)

    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        # print('.>>> Home team: ['+home_team+'] count: ['+str(count)+']')
        csv_output_array[count]['home_team_name'] = home_team
        csv_output_array[count]['away_team_name'] = away_team
        csv_output_array[count]['win_prediction_confidence'] = f"{winner_confidence}%"

        str_home_team = ""
        str_away_team = ""

        if winner == 1:
            team_game_rec['predicted_winner'] = str_home_team
            str_home_team += Style.RESET_ALL + Fore.GREEN + home_team.ljust(int_max_team_name_length)
            str_away_team += Style.RESET_ALL + Fore.RED + away_team.rjust(int_max_team_name_length + 8)

            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            str_home_team += Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)".rjust(8)
        else:
            team_game_rec['predicted_winner'] = str_away_team
            str_away_team += Style.RESET_ALL + Fore.GREEN + away_team.rjust(int_max_team_name_length)
            str_home_team += Style.RESET_ALL + Fore.RED + home_team.ljust(int_max_team_name_length + 8)

            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            str_away_team += Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)".rjust(8)

        str_uo = ""

        if under_over == 0:
            str_uo += (Style.RESET_ALL + Fore.MAGENTA + 'UNDER'.ljust(6) + Style.RESET_ALL
                       + str(todays_games_uo[count]).ljust(5))
            un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
            str_uo += Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)".rjust(8)
        else:
            str_uo += (Style.RESET_ALL + Fore.BLUE + 'OVER'.ljust(6) + Style.RESET_ALL
                       + str(todays_games_uo[count]).ljust(5))
            un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
            str_uo += Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)".rjust(8)

        str_vs = Style.RESET_ALL + ' vs '
        str_separator = Style.RESET_ALL + ': '

        print(Style.RESET_ALL + str_home_team + str_vs + str_away_team + str_separator + str_uo + Style.RESET_ALL)

        count += 1

    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))

            # print('>>> home team odds: ['+str(home_team_odds[count])+'] away team odds: ['+str(away_team_odds[count])+']')
        expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
                                 'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}
        bankroll_descriptor = ' Fraction of Bankroll: '
        bankroll_fraction_home = bankroll_descriptor + str(kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])) + '%'

        print(home_team.ljust(22) + ' EV: ' + expected_value_colors['home_color'] + str(ev_home).rjust(6) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team.ljust(22) + ' EV: ' + expected_value_colors['away_color'] + str(ev_away).rjust(6) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
        count += 1
    count = 0
    for team_game_rec in csv_output_array:

        print(csv_output_array[count]['home_team_odds'],
              csv_output_array[count]['home_team_name'],
              csv_output_array[count]['away_team_name'],
              csv_output_array[count]['away_team_odds'],
              csv_output_array[count]['predicted_winner'],
              csv_output_array[count]['win_prediction_confidence'],
              csv_output_array[count]['over_under_point'],
              csv_output_array[count]['over_under_prediction'],
              csv_output_array[count]['over_under_confidence'])
        print(count, csv_output_array[count])
        count += 1
        to_csv.append_to_csv(
            team_game_rec['home_team_odds'],
            team_game_rec['home_team_name'],
            team_game_rec['away_team_name'],
            team_game_rec['away_team_odds'],
            team_game_rec['predicted_winner'],
            team_game_rec['win_prediction_confidence'],
            team_game_rec['over_under_point'],
            team_game_rec['over_under_prediction'],
            team_game_rec['over_under_confidence'])

    deinit()
