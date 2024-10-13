import os
import shutil

import mlflow
import random

import yaml

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
        self.mlflow_dir_path = "C:\\Users\\david\\code\\NBA-Machine-Learning-Sports-Betting\\mlflow"

    def cleanup_under_performing_models(self):
        # Get the experiment ID
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        experiment_id = experiment.experiment_id

        # Fetch all runs in the experiment
        runs = mlflow.search_runs(experiment_ids=experiment_id)

        if 'metrics.accuracy' not in runs.columns:
            raise ValueError("Accuracy metric not found in runs.")

        # Calculate the bottom 50% threshold
        median_accuracy = runs['metrics.accuracy'].median()

        # Identify the runs to delete
        runs_to_delete = runs[runs['metrics.accuracy'] < median_accuracy]

        # Delete the bottom 50% performing runs
        for run_id in runs_to_delete['run_id']:
            if not self.is_run_in_saved_models(run_id):
                mlflow.delete_run(run_id)
                # Manually delete run files from the filesystem
                self.delete_ml_flow_artifacts(experiment_id=experiment_id, run_id=run_id)

    def is_run_in_saved_models(self, run_id: str) -> bool:
        """
        Checks if the run_id exists in any meta.yaml file under the models directory.
        """
        ml_runs_dir = os.path.join(self.mlflow_dir_path, 'mlruns')
        models_dir = os.path.join(ml_runs_dir, 'models')

        # Walk through the models directory and search for meta.yaml files
        for root, dirs, files in os.walk(models_dir):
            if 'meta.yaml' in files:
                meta_file_path = os.path.join(root, 'meta.yaml')
                with open(meta_file_path, 'r') as file:
                    meta_data = yaml.safe_load(file)

                    # If the run_id in the meta.yaml matches the current run_id, return True
                    if meta_data.get('run_id') == run_id:
                        return True

        # Return False if no matching run_id is found
        return False

    def delete_ml_flow_artifacts(self, experiment_id: str, run_id: str) -> None:
        artifact_dir = f"{self.mlflow_dir_path}\\mlartifacts\\{experiment_id}\\{run_id}"
        if os.path.exists(artifact_dir):
            shutil.rmtree(artifact_dir)
            print(f"Deleted run {run_id}")
        else:
            print(f"Run directory {artifact_dir} does not exist.")

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
