import mlflow
import numpy as np
import pandas as pd
import random
from Models.BaseModel import BaseModel
from data_cuts.base_data_cut import BaseDataCut


class ExperimentManager:
    def __init__(self):
        self.models = {}
        self.data_cuts = {}

    def add_model(self, model):
        if not isinstance(model, BaseModel):
            raise TypeError(TypeError("Parameter [model] must be of type BaseModel"))
        self.models[model.get_name()] = model

    def add_data_cut(self, data_cut):
        if not isinstance(data_cut, BaseDataCut):
            raise TypeError(TypeError("Parameter [data_cut] must be of type BaseDataCut"))
        self.data_cuts[data_cut.get_name()] = data_cut

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
