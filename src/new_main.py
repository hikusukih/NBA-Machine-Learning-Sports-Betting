import mlflow
import xgboost as xgb
from Predict.ModelManager import ModelManager
from Predict.XGBModel001 import XGBModel001
from ProcessData.DataProcessing import DataProcessor


class NNModel001:
    pass


def main():
    # Initialize DataProcessor
    data_processor = DataProcessor('nba_game_data.csv')
    data_processor.load_data()
    data_processor.preprocess_data()

    # Split the data
    X_train, X_test, y_train, y_test = data_processor.split_data()

    # Initialize ModelManager
    manager = ModelManager()

    # Add different models
    xgb_model = XGBModel001(params={'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1})
    nn_model = NNModel001(params={'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'alpha': 0.0001})

    manager.add_model(xgb_model)
    manager.add_model(nn_model)

    # Set up MLflow experiment
    mlflow.set_experiment("NBA Game Prediction")

    # Train all models
    manager.train_all_models(X_train, y_train)

    # Evaluate models
    results = manager.evaluate_models(X_test, y_test)

    # Print results
    for model_name, accuracy in results.items():
        print(f"{model_name} Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()