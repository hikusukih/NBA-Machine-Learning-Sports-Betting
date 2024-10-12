import mlflow
import numpy as np
import pandas as pd
import random
from Models.BaseModel import BaseModel
from data_cuts.base_data_cut import BaseDataCut
from experiments.model_data_training_run import ModelDataTrainingRun


class ExperimentManager:
    def __init__(self):
        self.models = {}
        self.data_cuts = {}
        self.experiments = {}
        self.experiment_name = "NBA Game Prediction"
        mlflow.set_tracking_uri("http://127.0.0.1:8765")
        mlflow.set_experiment(self.experiment_name)

    def cleanup_under_performing_models(self):
        # Get the experiment ID
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        experiment_id = experiment.experiment_id

        # Fetch all runs in the experiment
        runs = mlflow.search_runs(experiment_ids=experiment_id)

        if 'accuracy' not in runs.columns:
            raise ValueError("Accuracy metric not found in runs.")

        # Calculate the bottom 50% threshold
        median_accuracy = runs['accuracy'].median()

        # Identify the runs to delete
        runs_to_delete = runs[runs['accuracy'] < median_accuracy]

        # Delete the bottom 50% performing runs
        for run_id in runs_to_delete['run_id']:
            mlflow.delete_run(run_id)
            print(
                f"Deleted run: {run_id} with accuracy: {runs_to_delete[runs_to_delete['run_id'] == run_id]['accuracy'].values[0]}")

    def add_experiment(self, p_model, p_data_cut):
        """
        Add an experiment. The model and data_cut are available for exhaustive training.
        :param p_model:
        :param p_data_cut:
        :return:
        """
        self.add_model(p_model)
        self.add_data_cut(p_data_cut)

        str_experiment_name = p_model.get_name() + "_" + p_data_cut.get_name()
        ex = ModelDataTrainingRun(str_experiment_name)
        self.experiments[str_experiment_name] = ex

    def add_model(self, p_model):
        """
        Add a model to this experiment.
        It will be run against ALL available {@code data_cut}s.
        :param p_model:
        :return: None
        """
        if not isinstance(p_model, BaseModel):
            raise TypeError(TypeError("Parameter [p_model] must be of type BaseModel"))
        self.models[p_model.get_name()] = p_model

    def add_data_cut(self, p_data_cut):
        """
        Add a BaseDataCut to this experiment.
        It will be run against ALL available {@code BaseModel}s.
        :param p_data_cut:
        :return: None
        """
        if not isinstance(p_data_cut, BaseDataCut):
            raise TypeError(TypeError("Parameter [data_cut] must be of type BaseDataCut"))
        self.data_cuts[p_data_cut.get_name()] = p_data_cut

    def train_and_evaluate_defined_experiments(self):
        """Iterate through the experiments and run them all

        implement next...

        :return:
        """
        pass

    def train_and_evaluate_all_combinations(self):
        # Shuffle model indices
        model_indices = list(self.models.values())
        random.shuffle(model_indices)

        # Shuffle data cut indices
        data_cut_indices = list(self.data_cuts.values())
        random.shuffle(data_cut_indices)

        results = {}

        for model in model_indices:
            accuracy_list = []
            for dc in data_cut_indices:
                with mlflow.start_run(run_name=model.get_name()):
                    model.train(dc.get_x_train_data(), dc.get_y_train_data())
                    predictions = model.predict(dc.get_x_test_data())

                    accuracy = (predictions == dc.get_y_test_data()).mean()
                    print(accuracy.shape)
                    accuracy_list.append(accuracy)
                    mlflow.log_metric("accuracy", accuracy)

                    model.log_mlflow()
                    dc.log_mlflow()
                    mlflow.end_run()

            average_accuracy = sum(accuracy_list) / len(accuracy_list)
            # mlflow.log_metric("average_accuracy", average_accuracy)
            print("Average accuracy:", average_accuracy)

            results[model.get_name()] = average_accuracy

        return results
