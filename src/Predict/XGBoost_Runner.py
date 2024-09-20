import copy
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc
from src.Utils import export as to_csv
from src.Utils import ExportMoneyline


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/ChosenModel/XGBoost_ML-4.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/ChosenModel/XGBoost_UO-9.json')
int_max_team_name_length = 22

def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion,
               p_date=""):
    if p_date == "":
        p_date = datetime.today().strftime("%Y-%m-%d")

    ml_predictions_array = []
    team_game_rec = {'home_team_odds': None, 'home_team_name': None, 'home_team_win_confidence' : None,
                     'away_team_name': None, 'away_team_odds': None, 'away_team_win_confidence' : None,
                     'predicted_winner': None,
                     'kelly_percentage_home': None, 'kelly_percentage_away': None,
                     'over_under_point': None,
                     'over_under_prediction': None, 'under_over_confidence': None}

    csv_output_array = [team_game_rec.copy() for _ in range(len(data))]

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
        # print(f'>>>XGBoost_Runner>>> game: {game}')
        # print(f'>>>XGBoost_Runner>>> data[count]: {data[count]}')
        # print(f'>>>XGBoost_Runner>>> ml_predictions_array[count]: {ml_predictions_array[count]}')
        csv_output_array[count]['home_team_odds'] = home_team_odds[count]
        csv_output_array[count]['home_team_name'] = game[0]
        csv_output_array[count]['away_team_name'] = game[1]
        csv_output_array[count]['away_team_odds'] = away_team_odds[count]
        # csv_output_array[count]['predicted_winner'] = ''
        csv_output_array[count]['over_under_point'] = todays_games_uo[count]
        csv_output_array[count]['over_under_prediction'] = ''
        # csv_output_array[count]['under_over_confidence'] = ''

        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence_vector = ml_predictions_array[count]

        str_home_team = ""
        str_away_team = ""

        if winner == 1:
            csv_output_array[count]['predicted_winner'] = home_team
            str_home_team += Style.RESET_ALL + Fore.GREEN + home_team.ljust(int_max_team_name_length)
            str_away_team += Style.RESET_ALL + Fore.RED + away_team.rjust(int_max_team_name_length + 8)
            winner_confidence = round(winner_confidence_vector[0][1] * 100, 1)
            loser_confidence = round(winner_confidence_vector[0][0] * 100, 1)
            csv_output_array[count]['home_team_win_confidence'] = winner_confidence
            csv_output_array[count]['away_team_win_confidence'] = loser_confidence
            str_home_team += Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)".rjust(8)
        else:
            csv_output_array[count]['predicted_winner'] = away_team
            str_away_team += Style.RESET_ALL + Fore.GREEN + away_team.rjust(int_max_team_name_length)
            str_home_team += Style.RESET_ALL + Fore.RED + home_team.ljust(int_max_team_name_length + 8)

            winner_confidence = round(winner_confidence_vector[0][0] * 100, 1)
            csv_output_array[count]['away_team_win_confidence'] = winner_confidence
            loser_confidence = round(winner_confidence_vector[0][1] * 100, 1)
            csv_output_array[count]['home_team_win_confidence'] = loser_confidence
            str_away_team += Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)".rjust(8)

        str_uo = ""

        if under_over == 0:
            str_uo += (Style.RESET_ALL + Fore.MAGENTA + 'UNDER'.ljust(6) + Style.RESET_ALL
                       + str(todays_games_uo[count]).ljust(5))
            un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
            csv_output_array[count]['under_over_confidence'] = un_confidence
            csv_output_array[count]['over_under_prediction'] = 'UNDER'
            str_uo += Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)".rjust(8)
        else:
            str_uo += (Style.RESET_ALL + Fore.BLUE + 'OVER'.ljust(6) + Style.RESET_ALL
                       + str(todays_games_uo[count]).ljust(5))
            un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
            csv_output_array[count]['under_over_confidence'] = un_confidence
            csv_output_array[count]['over_under_prediction'] = 'OVER'
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
        else:
            continue
            # print('>>> home team odds: ['+str(home_team_odds[count])+'] away team odds: ['+str(away_team_odds[count])+']')
        expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
                                 'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}
        bankroll_descriptor = ' Fraction of Bankroll: '

        kelly_home_team = kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])
        kelly_away_team = kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])

        csv_output_array[count]['kelly_percentage_home'] = kelly_home_team
        csv_output_array[count]['kelly_percentage_away'] = kelly_away_team

        bankroll_fraction_home = bankroll_descriptor + str(kelly_home_team) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(kelly_away_team) + '%'

        print(home_team.ljust(22) + ' EV: ' + expected_value_colors['home_color'] + str(ev_home).rjust(6) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team.ljust(22) + ' EV: ' + expected_value_colors['away_color'] + str(ev_away).rjust(6) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
        count += 1

    for team_game_rec in csv_output_array:
        ExportMoneyline.append_to_csv(
            p_team_name=team_game_rec['home_team_name'],
            p_win_prediction_confidence=team_game_rec['home_team_win_confidence'],
            p_current_odds=team_game_rec['home_team_odds'],
            p_game_date=p_date
        )
        ExportMoneyline.append_to_csv(
            p_team_name=team_game_rec['away_team_name'],
            p_win_prediction_confidence=team_game_rec['away_team_win_confidence'],
            p_current_odds=team_game_rec['away_team_odds'],
            p_game_date=p_date
        )
        to_csv.append_to_csv(
            team_game_rec['home_team_odds'],
            team_game_rec['home_team_name'],
            "",
            "",
            team_game_rec['predicted_winner'],
            team_game_rec['home_team_win_confidence'],
            team_game_rec['over_under_point'],
            team_game_rec['over_under_prediction'],
            team_game_rec['under_over_confidence'],
            team_game_rec['kelly_percentage_home'])
        to_csv.append_to_csv(
            "",
            "",
            team_game_rec['away_team_name'],
            team_game_rec['away_team_odds'],
            team_game_rec['predicted_winner'],
            team_game_rec['away_team_win_confidence'],
            team_game_rec['over_under_point'],
            team_game_rec['over_under_prediction'],
            team_game_rec['under_over_confidence'],
            team_game_rec['kelly_percentage_away'])

    deinit()
