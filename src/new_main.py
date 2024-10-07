import mlflow
import time

from Models.MoneyLineNeuralNetModel001 import MoneyLineNeuralNetModel001
from Models.XGBModel001 import XGBModel001
from Models.NNModel001 import NNModel001
from ProcessData.InitialDataProcessor import InitialDataProcessor
from data_cuts.all_available_data import AllAvailableData
from experiments.experiment_manager import ExperimentManager

# from data_cuts.all_available_data import AllAvailableData


def main():
    # Initialize DataProcessor
    data_processor = InitialDataProcessor("../Data/dataset.sqlite")
    data_processor.load_data()
    data_processor.preprocess_data()

    # define data_cuts
    dc_all_data = AllAvailableData(feature_data=data_processor.X,
                                   label_data=data_processor.y,
                                   random_state=42,
                                   test_size=0.2)

    # define models
    xgb_model = XGBModel001(params={'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1})
    nn_model = NNModel001(params={'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'alpha': 0.0001})
    og_nn_model = MoneyLineNeuralNetModel001(params={'data_cut_name': 'All Data'})

    # Initialize Manager
    mgr = ExperimentManager()

    data_cuts = {}
    data_cuts[dc_all_data.get_name()] = dc_all_data

    models = {}
    models[xgb_model.get_name()] = xgb_model
    models[nn_model.get_name()] = nn_model
    models[og_nn_model.get_name()] = og_nn_model

    # Consider iterating all models/datacuts and adding to the ExperimentManager
    # but for now...
    mgr.add_model(xgb_model)
    mgr.add_model(nn_model)
    mgr.add_model(og_nn_model)
    mgr.add_data_cut(dc_all_data)

    print("Begin training:")
    # Train all models
    results = mgr.train_and_evaluate_all_combinations()
    print("Training complete.")

    # Print results (manager should have transitively logged them anyway)
    for model_name, accuracy in results.items():
        print(f"{model_name} Average Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
