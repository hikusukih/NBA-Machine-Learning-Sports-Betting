import mlflow
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from Models.BaseModel import BaseModel


class MoneyLineNeuralNetModel001(BaseModel):
    """
    The MoneyLine model that came with the code
    """
    def __init__(self,params):
        self.model_name = 'MoneyLineNeuralNetModel001'
        self.model = None
        self.params = params

    def train(self, x_train, y_train):
        with mlflow.start_run(run_name=self.model_name):
            current_time = str(time.time())

            tensorboard = TensorBoard(log_dir='../../Logs/{}'.format(current_time))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

            model_path = '../../Models/NN_Models/'
            # model_filename = 'Trained-Model-MoneyLine-' + current_time + '.kerasext'
            model_best_accuracy_filename = model_path + 'Model-MoneyLine-Most-Accuracy.h5'
            model_least_loss_filename = model_path + 'Model-MoneyLine-Least-Loss.h5'

            # mcp_save = ModelCheckpoint(
            #     model_path + model_filename,
            #     save_best_only=True, monitor='val_loss', mode='min')

            # Checkpoint for the model with the best accuracy
            accuracy_checkpoint = ModelCheckpoint(
                model_path + model_best_accuracy_filename,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )

            # Checkpoint for the model with the lowest loss
            loss_checkpoint = ModelCheckpoint(
                model_path + model_least_loss_filename,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
)

            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)),
                tf.keras.layers.Dense(512, activation=tf.nn.relu6),
                tf.keras.layers.Dense(256, activation=tf.nn.relu6),
                tf.keras.layers.Dense(128, activation=tf.nn.relu6),
                tf.keras.layers.Dense(2, activation=tf.nn.softmax),
            ])

            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            model_summary_str = "\n".join(model_summary)
            with open("model_summary.txt", "w") as f:
                f.write(model_summary_str)
            mlflow.log_artifact("model_summary.txt")  # Log the summary as an artifact

            # Log parameters
            mlflow.log_params(self.params)  # Log initial parameters
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("batch_size", 32)  # Example fixed parameter
            mlflow.log_param("epochs", 50)  # Example fixed parameter

            # Train the model with logging
            history = self.model.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=32,
                           callbacks=[tensorboard, early_stopping, accuracy_checkpoint, loss_checkpoint])

            # Log metrics for each epoch (training and validation loss/accuracy)
            for epoch in range(len(history.history['loss'])):
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

            # Save the final model as an artifact
            self.model.save(f"final_model_{self.model_name}.h5")
            mlflow.log_artifact(f"final_model_{self.model_name}.h5")
            mlflow.log_artifact(model_least_loss_filename)
            mlflow.log_artifact(model_best_accuracy_filename)
    def predict(self, x_test):
        print( x_test.shape)
        return self.model.predict(x_test)

    def log_model(self):
        mlflow.sklearn.log_model(self.model, self.model_name)
