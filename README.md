# NBA Sports Betting Using Machine Learning 🏀
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/output.png" width="1010" height="292" />

A machine learning AI used to predict the winners and under/overs of NBA games. 
Takes all team data from the 2007-08 season to current season, matched with odds of those games, 
using a neural network to predict winning bets for today's games. 

Achieves ~69% accuracy on money lines and ~55% on under/overs. 

Outputs expected value for teams money lines to provide better insight. The fraction of your bankroll to bet based on 
the Kelly Criterion is also outputted.

Note that a popular, less risky approach is to bet 50% of the stake recommended by the Kelly Criterion.
## Packages Used

Use Python 3.11. In particular the packages/libraries used are...

* Tensorflow - Machine learning library
* XGBoost - Gradient boosting framework
* Numpy - Package for scientific computing in Python
* Pandas - Data manipulation and analysis
* Colorama - Color text output
* Tqdm - Progress bars
* Requests - Http library
* Scikit_learn - Machine learning library

## Usage

<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Expected_value.png" width="1010" height="424" />

Make sure all packages above are installed.

```bash
$ git clone https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting.git
$ cd NBA-Machine-Learning-Sports-Betting
$ pip3 install -r requirements.txt
$ python main.py -xgb -odds=fanduel
```

Odds data will be automatically fetched from sbrodds if the -odds option is provided with a sportsbook.  
Options include: fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny

If `-odds` is not given, enter the under/over and odds for today's games manually after starting the script.

Optionally, you can add '-kc' as a command line argument to see the recommended fraction of your bankroll to wager based on the model's edge

## Flask Web App
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Flask-App.png" width="922" height="580" />

This repo also includes a small Flask application to help view the data from this tool in the browser.  To run it:
```
cd Flask
flask --debug run
```

## Overview
- Get new data
  - get_data
    - Get stats from stats.nba.com, day by day
  - get_odds_data
  - create_games
- Train Models
  - Money-Line
  - Under-Over

## Getting new data and training models
```
# Create dataset with the latest data for 2023-24 season
cd src/Process-Data
python -m Get_Data; python -m Get_Odds_Data; python -m Create_Games

# Train models
cd ../Train-Models
python -m XGBoost_Model_ML
python -m XGBoost_Model_UO
```

## Contributing

All contributions welcomed and encouraged.

## Notes:
Consuming a model may require you to be on the same version of Keras/Tensorflow as the one on which the model was created.

TQDM: Adds a progress bar to iterables in the console output

### Workflow for a day:
- Get the latest win/loss data (TODO: Append!)
- Train on the latest win/loss data
- Run the model on today's games
- Move the data to the Spreadsheet and figure out the betting strategy
- Bet!
### Strategy
- Bet x% of bankroll in a given day (X<100) - the max you're comfortable completely losing
- Old:
  - bet games with 70+% probability
  - Apportion based on relative confidence
- New:
  - bet based on Kelly Criterion
    - if the total suggested bankroll percentage sums to less than 100, use the suggested percentage
    - Otherwise, weight the suggested percentages against one another to not exceed the max bankroll
- Any 0 KC gets the same rank as the lowest value (so i always bet every one)

# Open Terminals
- **RunModel**
  - $ python main.py -xgb -odds=fanduel 
    - `-t` for "tomorrow"
  - $ python -m Get_Data; python -m Get_Odds_Data; python -m Create_Games
- **Hyperparam Search**
  - cd src/Train-Models ; python Hyperparam_XGBoost_ML.py ;
- **JupyterLab**
  - py -m jupyterlab