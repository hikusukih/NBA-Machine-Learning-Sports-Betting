import mlflow
import numpy as np
import pandas as pd
import random
from Models.BaseModel import BaseModel
from data_cuts.base_data_cut import BaseDataCut
from experiments.data_model_experiment import ModelDataTrainingRun


class ExperimentManager:
    def __init__(self):
        self.models = {}
        self.data_cuts = {}
        self.experiments = {}

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
        """
        Iterate through the experiments and run them all
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
                model.train(dc.get_x_train_data(), dc.get_y_train_data())
                predictions = model.predict(dc.get_x_test_data())
                model.log_model()

                accuracy = (predictions == dc.get_y_test_data()).mean()
                print(accuracy.shape)
                accuracy_list.append(accuracy)
                # results[model.get_name() + ] = accuracy
                mlflow.log_metric(f"accuracy_{model.get_name()}_{dc.get_name()}", accuracy)
                mlflow.end_run()

            average_accuracy = sum(accuracy_list) / len(accuracy_list)
            # mlflow.log_metric("average_accuracy", average_accuracy)
            print("Average accuracy:", average_accuracy)

            results[model.get_name()] = average_accuracy

        return results
