import os

import mlflow
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from Models.BaseModel import BaseModel


class MoneyLineNeuralNetModel002(BaseModel):
    """
    The MoneyLine model that I tweak!
    """
    def __init__(self,params=None):
        super().__init__('MoneyLineNeuralNetModel002')
        self.history = None
        self.params = params

    def train(self, x_train, y_train):
        current_time = str(time.time())

        tensorboard = TensorBoard(log_dir='../../Logs/{}'.format(current_time))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

        model_output_path = '../../Models/NN_Models/'
        # model_filename = 'Trained-Model-MoneyLine-' + current_time + '.kerasext'
        model_best_accuracy_filename = model_output_path + 'Model-MoneyLine-Most-Accuracy.h5'
        model_least_loss_filename = model_output_path + 'Model-MoneyLine-Least-Loss.h5'

        # mcp_save = ModelCheckpoint(
        #     model_output_path + model_filename,
        #     save_best_only=True, monitor='val_loss', mode='min')

        # Checkpoint for the model with the best accuracy
        accuracy_checkpoint = ModelCheckpoint(
            model_best_accuracy_filename,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )

        # Checkpoint for the model with the lowest loss
        loss_checkpoint = ModelCheckpoint(
            model_least_loss_filename,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu6),
            tf.keras.layers.Dropout(0.5),  # Dropout layer added
            tf.keras.layers.Dense(256, activation=tf.nn.relu6),
            tf.keras.layers.Dropout(0.5),  # Dropout layer added
            tf.keras.layers.Dense(128, activation=tf.nn.relu6),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax),
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model_summary = []
        self.model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary_str = "\n".join(model_summary)
        print(os.getcwd())
        model_summary_path = "../out/temp/model_summary.txt"
        with open(model_summary_path, "w") as f:
            f.write(model_summary_str)
        print(os.path.abspath(f.name))
        mlflow.log_text(model_summary_str, artifact_file=os.path.abspath(f.name))  # Log the summary as an artifact

        # Log parameters
        mlflow.log_param("model_name", self.model_name)
        mlflow.log_param("batch_size", 32)  # Example fixed parameter
        mlflow.log_param("epochs", 50)  # Example fixed parameter

        # Train the model, logging along the way
        self.history = self.model.fit(x_train, y_train,
                                      epochs=50, validation_split=0.1, batch_size=32,
                                      callbacks=[tensorboard, early_stopping,
                                                 accuracy_checkpoint, loss_checkpoint])

        # Save the final model as an artifact
        model_final_filename = f"../out/temp/final_model_{self.model_name}.h5"

        self.model.save(model_final_filename)
        mlflow.log_artifact(model_final_filename)
        mlflow.log_artifact(model_least_loss_filename)
        mlflow.log_artifact(model_best_accuracy_filename)

    def predict(self, x_test):
        return np.argmax(self.model.predict(x_test), axis=1)

    def log_mlflow(self):
        # Log the model itself as a Keras model
        mlflow.keras.log_model(self.model, self.model_name)
        # mlflow.sklearn.log_model(self.model, self.model_name)
        # Log metrics for each epoch (training and validation loss/accuracy)
        for epoch in range(len(self.history.history['loss'])):
            mlflow.log_metric("train_loss", self.history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", self.history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", self.history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", self.history.history['val_accuracy'][epoch], step=epoch)
