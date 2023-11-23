import argparse
from typing import Dict, Optional, Tuple
from pathlib import Path
import flwr as fl
import tensorflow as tf
from glob import glob
import numpy as np
from ast import literal_eval
from data import dataLoader, sampledDataBuilder
from model import architecture
from optimizer import dpSGD
from hydra import compose, initialize
from omegaconf import OmegaConf
from conf import logConfig
from utils import csvWriter, AWSwriter
from mlflow import log_metric, log_param, log_params, log_artifacts
import uuid
import os
from datetime import datetime





def main(logger, cfg, id_, record, myuuid) -> None:

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation

    # push the config file to s3/ dynamo db

    # s3writer.push_object_to_s3(f"{myuuid}.json",
    #                             cfg['AWS']['config']['bucket_name'],
    #                             dict(cfg['data']))

    model = architecture.DriverIdentification(n_timesteps=cfg['model']['n_timesteps'],
                                            n_features=cfg['model']['n_features'],
                                            n_outputs=cfg['model']['n_outputs'],
                                            window_size=cfg['data']['window_size'])
    logger.info(f" {id_} Model Architecture is initialized")
    logger.info(f" {id_} {model.summary()}")

    if cfg['training']['optimizer'] == 'DPSGD':
        opt = dpSGD.DifferentiallyPrivateSGD(l2_norm_clip=cfg['dpSGD']['l2_norm_clip'],
                                            noise_multiplier=cfg['dpSGD']['noise_multiplier'],
                                            num_microbatches=cfg['dpSGD']['num_microbatches'],
                                            learning_rate=cfg['dpSGD']['learning_rate'])
    elif cfg['training']['optimizer'] == 'SGD':
        opt = tf.keras.optimizers.SGD(cfg[cfg['training']['optimizer']]['learning_rate'])

    logger.info(f" {id_} {opt} Optimizer is used")
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss= loss,
                  optimizer=opt,
                  metrics=[cfg['training']['metrics']])
    logger.info(f" {id_}  Model is compiled")

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
            fraction_fit= cfg['server']['strategy']['fraction_fit'],
            fraction_evaluate= cfg['server']['strategy']['fraction_evaluate'],
            min_fit_clients= cfg['server']['strategy']['min_fit_clients'],
            min_evaluate_clients= cfg['server']['strategy']['min_evaluate_clients'],
            min_available_clients= cfg['server']['strategy']['min_available_clients'],
            evaluate_fn=get_evaluate_fn(model, cfg, id_, record, myuuid),
            on_fit_config_fn=get_on_fit_config_fn(cfg, myuuid),
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
            server_address=f"{cfg['server']['ip']}:{cfg['server']['port']}",
            config=fl.server.ServerConfig(num_rounds= cfg['server']['training']['num_rounds']),
            strategy=strategy
    )


def get_evaluate_fn(model, cfg,  id_, record, myuuid):
    """Return an evaluation function for server-side evaluation."""


    log_params(cfg['data'])
    log_params(cfg['data_distribution'])
    log_params(cfg['training'])
    log_params(cfg['dpSGD'])
    log_params(cfg['server'])
    log_params(cfg['privacy_accountant'])

    logger.info(f"Execution of {id_}")
    logger.info(f"Execution Configuration {cfg}")

    window_size = cfg['data']['window_size']
    number_of_samples = cfg['data']['number_of_samples']
    verbose = cfg['training']['verbose']
    epochs = cfg['training']['epochs']
    n_timesteps = cfg['model']['n_timesteps']
    n_features = cfg['model']['n_features']
    n_outputs = cfg['model']['n_outputs']
    batch_size = cfg['model']['batch_size']

    train_df, test_df, label_encoder = sampledDataBuilder.data_split(id_,
                                                                    record,
                                                                    cfg,
                                                                    logger)
    logger.info(f" {id_} Data split completed")

    test_window_sequences = sampledDataBuilder.create_test_window_sequences(test_df,
                                                                            label_encoder,
                                                                            window_size,
                                                                            number_of_samples,
                                                                            logger,
                                                                            id_)
    logger.info(f" {id_} Test window sequences generated")

    test_gen = dataLoader.DriverDatasetGenerator(test_window_sequences,
                                       batch_size,
                                       n_features,
                                       n_timesteps,
                                       n_outputs)
    logger.info(f" {id_} Train generator is initialized")


    # The `evaluate` function will be called after every round
    def evaluate(server_round: int,
                parameters: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(test_gen, steps = len(test_gen)//batch_size)
        log_metric("loss", loss)
        log_metric("accuracy", accuracy)

        # id_, myuuid, server_round, loss, accuracy, test_gen_len
        csv_write_status = csvWriter.write_server_results_to_csv(id_,
                                              myuuid,
                                              server_round,
                                              loss,
                                              accuracy,
                                              len(test_gen))

        if csv_write_status and \
            server_round == cfg['server']['training']['num_rounds']:
            """
            push this csv to S3 bucket
            """
            s3_write_status = AWSwriter.push_data_to_s3(f"{id_}_{myuuid}.csv",
                                              cfg['AWS']['trainingCSV']['bucket_name'],
                                              f"./csvOutputs/{id_}_{myuuid}.csv")
            if s3_write_status:
                os.remove(f"./csvOutputs/{id_}_{myuuid}.csv")


        return  loss, {"eval records": len(test_gen), "accuracy": accuracy}

    return evaluate



def get_on_fit_config_fn(cfg, myuuid):

    def fit_config(server_round: int):
            """Return training configuration dict for each round.

            Keep batch size fixed at 32, perform two rounds of training with one
            local epoch, increase to two local epochs afterwards.
            """

            config = {
                    "batch_size": cfg['model']['batch_size'],
                    "local_epochs": cfg['server']['training']['client_first_epoch_count'] if server_round < 2 else cfg['server']['training']['client_later_epoch_count'],
                    "server_round": server_round,
                    "myuuid": myuuid
            }

            return config

    return fit_config


def evaluate_config(server_round: int):
        """Return evaluation configuration dict for each round.

        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds one to three, then increase to ten local
        evaluation steps.
        """
        val_steps = 5 if server_round < 4 else 10
        return {"val_steps": val_steps}




if __name__ == "__main__":

        id_ = "server"
        record = 'evaluation_record'

        logger = logConfig.get_logger()
        myuuid = str(uuid.uuid1())

        # config
        initialize(version_base= "1.1", config_path="./conf", job_name="client_app")
        cfg = compose(config_name="flConfig")
        np.random.seed(cfg['data_distribution']['seed'])
        main(logger, cfg, id_, record, myuuid)
