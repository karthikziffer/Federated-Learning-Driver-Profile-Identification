import argparse
import os
from pathlib import Path

import tensorflow as tf
import flwr as fl
import numpy as np
import keras
from glob import glob
from ast import literal_eval
import datetime
from hydra import compose, initialize
from omegaconf import OmegaConf
from data import dataLoader, sampledDataBuilder
from model import architecture
from optimizer import dpSGD
from conf import logConfig
from utils import csvWriter, callbacks, AWSwriter
from datetime import datetime
from mlflow import log_metric, log_param, log_params, log_artifacts
import mlflow.keras 
import sys
import json
import decimal



# Define Flower client
class AutomobileClient(fl.client.NumPyClient):
        def __init__(self, model, traingen, valgen, logger, client_id, cfg, dynamo_table):
                self.model = model
                self.traingen = traingen
                self.valgen = valgen
                self.logger = logger
                self.client_id = client_id
                self.cfg = cfg
                self.dynamo_table = dynamo_table

        def get_properties(self, config):
                """Get properties of client."""
                raise Exception("Not implemented")

        def get_parameters(self, config):
                """Get parameters of the local model."""
                raise Exception("Not implemented (server-side parameter initialization)")

        def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local model parameters
                self.model.set_weights(parameters)

                # Get hyperparameters for this round
                batch_size: int = config["batch_size"]
                epochs: int = config["local_epochs"]

                datetime_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                log_dir = f"./logs/{self.client_id}/{datetime_id}"  
                file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
                file_writer.set_as_default()

                # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

                # model checkpoint
                model_checkpoint = tf.keras.callbacks.ModelCheckpoint(f"./modelCheckpoints/{self.client_id}_{config['myuuid']}.h5",
                                                    monitor = "accuracy",
                                                    verbose  = 1,
                                                    save_best_only= True,
                                                    save_weights_only = False,
                                                    mode = "max",
                                                    save_freq ="epoch"
                                                    )

                # Train the model using hyperparameters from config
                history = self.model.fit_generator(self.traingen,  
                                                    epochs = epochs, 
                                                    steps_per_epoch = len(self.traingen)//batch_size,
                                                    validation_steps = len(self.valgen)//batch_size,
                                                    validation_data=self.valgen,
                                                    callbacks = [
                                                    model_checkpoint,
                                                    callbacks.CustomCallback(self.cfg['privacy_accountant']['number_of_examples'], 
                                                        self.cfg['privacy_accountant']['batch_size'], 
                                                        self.cfg['privacy_accountant']['noise_multiplier'],
                                                        self.cfg['privacy_accountant']['delta'],
                                                        self.cfg['privacy_accountant']['used_microbatching'], 
                                                        logger, 
                                                        log_metric, 
                                                        self.client_id,
                                                        config, 
                                                        self.cfg, 
                                                        self.dynamo_table)])


                # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
                # gradient_norm = 0
                # counter = 0
                # for  X, y in self.traingen:
                #     if counter < len(self.traingen)//batch_size:
                #         with tf.GradientTape() as tape:
                #             logits = self.model(X)
                #             loss_value = loss_fn(y, logits)
                #         grads = tape.gradient(loss_value, self.model.trainable_weights)
                #         gradient_norm += tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in grads]))
                #         counter += 1

                # log_metric("gradient_norm", gradient_norm)


                # Return updated model parameters and results
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.traingen)
                results = {
                        "loss": history.history["loss"][0],
                        "accuracy": history.history["accuracy"][0],
                        "val_loss": history.history["val_loss"][0],
                        "val_accuracy": history.history["val_accuracy"][0],
                        "train_records":  len(self.traingen), 
                        "test_records": len(self.valgen),  
                        "batch_size": batch_size, 
                        "epochs": epochs
                }


                log_metric("loss", history.history["loss"][0])
                log_metric("accuracy", history.history["accuracy"][0])
                log_metric("val_loss", history.history["val_loss"][0])
                log_metric("val_accuracy", history.history["val_accuracy"][0])
                log_metric("server_round", config["server_round"])

                self.logger.info(f" {self.client_id} training results: {results}")

                tf.summary.scalar('Train Accuracy', 
                                    data=history.history["accuracy"][0], 
                                    step=config["server_round"])



                """
                the code control comes here, checks if the csv exists, 
                if not creates a csv and writes the row
                if exists, just write the row 
                """

                csv_write_status = csvWriter.write_client_results_to_csv(self.client_id, 
                                                      config['myuuid'],
                                                      self.cfg,
                                                      results, 
                                                      config)

                if csv_write_status and \
                    config["server_round"]==self.cfg['server']['training']['num_rounds']:
                    """
                    push this csv to S3 bucket
                    """
                    s3_write_status = AWSwriter.push_data_to_s3(f"{self.client_id}_{config['myuuid']}.csv", 
                                                      self.cfg['AWS']['trainingCSV']['bucket_name'], 
                                                      f"./csvOutputs/{self.client_id}_{config['myuuid']}.csv")
                    if s3_write_status:
                        os.remove(f"./csvOutputs/{self.client_id}_{config['myuuid']}.csv")


                    """ write model to s3 bucket """
                    s3_model_write_status = AWSwriter.push_data_to_s3(f"{self.client_id}_{config['myuuid']}.h5", 
                                                      self.cfg['AWS']['models']['bucket_name'], 
                                                      f"./modelCheckpoints/{self.client_id}_{config['myuuid']}.h5")

                    if s3_model_write_status:
                        os.remove(f"./modelCheckpoints/{self.client_id}_{config['myuuid']}.h5")



                return parameters_prime, num_examples_train, results

        def evaluate(self, parameters, config):
                """Evaluate parameters on the locally held test set."""

                # Update local model with global parameters
                self.model.set_weights(parameters)

                # Get config values
                # steps: int = config["val_steps"]

                # Evaluate global model parameters on the local test data and return results
                loss, accuracy = self.model.evaluate(self.valgen, steps= len(self.valgen)//cfg['model']['batch_size'])
                num_examples_test = len(self.valgen)
                return loss, num_examples_test, { "accuracy": accuracy}


def main(logger, cfg, client_id, record) -> None:

    try:

        log_params(cfg['data'])
        log_params(cfg['data_distribution'])
        log_params(cfg['training'])
        log_params(cfg['dpSGD'])
        log_params(cfg['server'])
        log_params(cfg['privacy_accountant'])
        logger.info(f"Execution of {client_id}")
        logger.info(f"Execution Configuration {cfg}")

        window_size = cfg['data']['window_size']
        number_of_samples = cfg['data']['number_of_samples']
        verbose = cfg['training']['verbose']
        epochs = cfg['training']['epochs']
        n_timesteps = cfg['model']['n_timesteps']
        n_features = cfg['model']['n_features']
        n_outputs = cfg['model']['n_outputs']
        batch_size = cfg['model']['batch_size']


        ############################################################################################

        # initialize the dynamodb table
        dynamo_table = AWSwriter.initialize_dynamo_db()

        ############################################################################################

        train_df, test_df, label_encoder = sampledDataBuilder.data_split(client_id, 
                                                                        record,
                                                                        cfg, 
                                                                        logger)
        logger.info(f" {client_id} Data split completed")

        train_window_sequences = sampledDataBuilder.create_train_window_sequences(train_df,
                                                                                label_encoder, 
                                                                                window_size, 
                                                                                number_of_samples, 
                                                                                logger, 
                                                                                client_id)
        logger.info(f" {client_id} Train window sequences generated")

        test_window_sequences = sampledDataBuilder.create_test_window_sequences(test_df, 
                                                                                label_encoder, 
                                                                                window_size, 
                                                                                number_of_samples, 
                                                                                logger, 
                                                                                client_id)
        logger.info(f" {client_id} Test window sequences generated")

        # Load and compile Keras model
        model = architecture.DriverIdentification(n_timesteps=n_timesteps, n_features=n_features, n_outputs=n_outputs, window_size=window_size)
        logger.info(f" {client_id} Model Architecture is initialized")
        logger.info(f" {client_id} {model.summary()}")

        if cfg['training']['optimizer'] == 'DPSGD':
            opt = dpSGD.DifferentiallyPrivateSGD(l2_norm_clip=cfg['dpSGD']['l2_norm_clip'], 
                                                noise_multiplier=cfg['dpSGD']['noise_multiplier'], 
                                                num_microbatches=cfg['dpSGD']['num_microbatches'], 
                                                learning_rate=cfg['dpSGD']['learning_rate'])
        elif cfg['training']['optimizer'] == 'SGD':
            opt = tf.keras.optimizers.SGD(cfg[cfg['training']['optimizer']]['learning_rate'])

        logger.info(f" {client_id} {opt} Optimizer is used")

        loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)

        model.compile(loss= loss, optimizer=opt, metrics=[cfg['training']['metrics']])
        logger.info(f" {client_id}  Model is compiled")


        train_gen = dataLoader.DriverDatasetGenerator(train_window_sequences,
                                           batch_size,
                                           n_features,
                                           n_timesteps,
                                           n_outputs)
        logger.info(f" {client_id} Train generator is initialized")

        test_gen = dataLoader.DriverDatasetGenerator(test_window_sequences,
                                           batch_size,
                                           n_features,
                                           n_timesteps,
                                           n_outputs)
        logger.info(f" {client_id} Test generator is initialized")


        # Start Flower client
        client = AutomobileClient(model, train_gen, test_gen, logger, client_id, cfg, dynamo_table)
        logger.info(f" {client_id} AutomobileClient is initialized")


        fl.client.start_numpy_client(
                server_address=f"{cfg['client']['ip']}:{cfg['client']['port']}",
                client=client
        )
        logger.info(f" {client_id} AutomobileClient is initialized")

    except Exception as e:
        logger.error(f" {client_id} Error occured and execution failed.  {e}")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)




if __name__ == "__main__":

        client_id = sys.argv[1]
        record = "training_record"

        logger = logConfig.get_logger()

        # config
        initialize(version_base= "1.1", config_path="./conf", job_name="client_app")
        cfg = compose(config_name="flConfig")

        np.random.seed(cfg['data_distribution']['seed'])

        # # Make TensorFlow logs less verbose
        # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        main(logger, cfg, client_id, record)

