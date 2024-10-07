import mlflow
import time
from Models.ModelManager import ModelManager
from Models.MoneyLineNeuralNetModel001 import MoneyLineNeuralNetModel001
from Models.XGBModel001 import XGBModel001
from Models.NNModel001 import NNModel001
from ProcessData.InitialDataProcessor import InitialDataProcessor
from data_cuts.all_available_data import AllAvailableData
from experiments.experiment_manager import ExperimentManager

# from data_cuts.all_available_data import AllAvailableData


def main():
    # Set up MLflow experiment
    mlflow.set_tracking_uri("http://127.0.0.1:8765")
    mlflow.set_experiment("NBA Game Prediction")

    # Initialize DataProcessor
    data_processor = InitialDataProcessor("../Data/dataset.sqlite")
    data_processor.load_data()
    data_processor.preprocess_data()

    # Split the data
    x_train, x_test, y_train, y_test = data_processor.split_data()
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    dc_all_data = AllAvailableData(feature_data=data_processor.X,
                                label_data=data_processor.y,
                                random_state=42,
                                test_size=0.2)

    print("Data split.")

    # Initialize ModelManager
    manager = ModelManager()
    mgr = ExperimentManager()

    data_cuts = {}
    data_cuts["all"] = AllAvailableData()

    models = {}


    # Add different models
    xgb_model = XGBModel001(params={'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1})
    nn_model = NNModel001(params={'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'alpha': 0.0001})
    og_nn_model = MoneyLineNeuralNetModel001(params={'data_cut_name': 'All Data'})

    models[xgb_model.get_name()] = xgb_model
    models[nn_model.get_name()] = nn_model
    models[og_nn_model.get_name()] = og_nn_model



    manager.add_model(og_nn_model)
    manager.add_model(xgb_model)
    manager.add_model(nn_model)


    print("Begin training:")
    # Train all models
    # manager.train_all_models(x_train, y_train)
    results = mgr.train_and_evaluate_all_combinations()
    print("Training complete.")

    print("Begin evaluation:")
    # Evaluate models
    # results = manager.evaluate_models(x_test, y_test)
    print("Evaluation complete.")

    # Print results
    for model_name, accuracy in results.items():
        print(f"{model_name} Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
