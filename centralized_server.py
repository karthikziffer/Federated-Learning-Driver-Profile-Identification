#Tensorflow
import tensorflow as tf
import keras

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

#other
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import math
import io
import datetime as dt
import re
from math import sqrt
import time 
from functools import reduce
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy_statement
import tensorflow_privacy
import re
from hydra import compose, initialize
from omegaconf import OmegaConf


import socket
import sys
import paho.mqtt.client as mqtt
import json
from ast import literal_eval
import uuid
from utils import csvWriter, callbacks, AWSwriter
import os 

try:
    import cPickle as pickle
except:
    import pickle


################################################################################################


class DriverDatasetGenerator(keras.utils.Sequence):

  def __init__(self, sequences, batch_size, n_channels, window_size, classes, shuffle=False):
    self.sequences = sequences
    self.batch_size = batch_size
    self.n_channels = n_channels
    self.window_size = window_size
    self.n_classes = classes
    self.shuffle = shuffle

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):

      if self.shuffle == True:
        np.random.shuffle(self.sequences)

      list_IDs_temp = [np.random.randint(len(self.sequences)) for _ in range(self.batch_size)]
      X, y = self.__data_generation(list_IDs_temp)
      return X, y

  def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

      X = np.empty((self.batch_size, self.window_size, self.n_channels))
      y = np.empty((self.batch_size, self.n_classes))

      # Generate data
      for i, _ in enumerate(list_IDs_temp):
          # Store sample
          X[i,], y[i,] = self.sequences[_]
      return X, y


##########################################################################################################################################
class CustomCallback(keras.callbacks.Callback):

    def __init__(self, number_of_examples, batch_size, noise_multiplier, delta, used_microbatching, model, train_gen):
        self.number_of_examples = number_of_examples
        self.batch_size = batch_size
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.used_microbatching = used_microbatching
        self.tmp_epoch = 0
        self.model = model
        self.train_gen = train_gen

    def on_epoch_end(self, epoch, logs=None):

        epoch += 1
        self.tmp_epoch += 1
        # calculate the privacy budget
        statement  = compute_dp_sgd_privacy_statement(number_of_examples = self.number_of_examples,
                                                      batch_size = self.batch_size,
                                                      num_epochs = epoch,
                                                      noise_multiplier = self.noise_multiplier,
                                                      delta = self.delta,
                                                      used_microbatching = self.used_microbatching)
        privacy_budget = re.search(r"epoch(:\d+.\d+)", statement.replace(" ", "")).group(0).split(':')[-1]

        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
        idx = 0
        gradient_norm = 0
        for  X, y in self.train_gen:
          idx += 1
          if idx < 10:
            with tf.GradientTape() as tape:
                logits = self.model(X)
                loss_value = loss_fn(y, logits)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            gradient_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in grads]))

################################################################################################


def on_connect(client, userdata, flags, rc):
  global cfg  
  print("Connected with result code "+str(rc))
  client.subscribe(cfg['centralized_learning']['mqtt_broker']['topic'])

def on_message(client, userdata, msg):
    print("message received")
    
    global cfg, train_data, test_data, counter, client_set

 
    message = msg.payload
    client_data = pickle.loads(msg.payload)
    cid = client_data['client_id']
  
    if cid not in client_set:
      if train_data and test_data:
        train_data = client_data["train_window_sequences"]
        test_data = client_data["test_window_sequences"]
      else:
        train_data = train_data + client_data["train_window_sequences"]
        test_data = test_data + client_data["test_window_sequences"]
      client_set.add(cid)
      counter += 1
      print(f"Counter on_message: {counter}")
      print(cid)
    
    
    if counter == 3:
      
      myuuid = str(uuid.uuid1())
      window_size = cfg['centralized_learning']['data']['window_size']
      number_of_samples = cfg['centralized_learning']['data']['number_of_samples']
      verbose = cfg['centralized_learning']['training']['verbose']
      epochs = cfg['centralized_learning']['training']['epochs']
      n_timesteps = window_size
      n_features = cfg['centralized_learning']['model']['n_features']
      n_outputs = cfg['centralized_learning']['model']['n_outputs']
      batch_size = cfg['centralized_learning']['model']['batch_size']

      # combine all the data from different clients

      client_id = client_data["client_id"]
      train_window_sequences = train_data
      test_window_sequences = test_data

      train_gen = DriverDatasetGenerator(train_window_sequences,
                                         batch_size,
                                         n_features,
                                         n_timesteps,
                                         n_outputs)

      test_gen = DriverDatasetGenerator(test_window_sequences,
                                         batch_size,
                                         n_features,
                                         n_timesteps,
                                         n_outputs)

      ################################################################################################

      # logger callback

      csv_folder_path = cfg['centralized_learning']['output_path']['csv']
      csv_logger = tf.keras.callbacks.CSVLogger(f'{csv_folder_path}{myuuid}_centralizedLearning.csv', 
                                                separator=",", 
                                                append=False)

      # model checkpoint callback
      model_checkpoint_folder_path = cfg['centralized_learning']['output_path']['model']
      model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        f'{model_checkpoint_folder_path}{myuuid}_centralizedLearning.h5',
        monitor = "val_accuracy",
        verbose = 1,
        save_best_only = True,
        save_weights_only = False,
        mode = "max",
        save_freq="epoch"
      )

      ################################################################################################

      model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                            strides=1,
                            activation="relu",
                            input_shape=(n_timesteps,n_features)),
        tf.keras.layers.LSTM(16, return_sequences =True),
        tf.keras.layers.LSTM(8, return_sequences =True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(n_outputs, activation="softmax")
      ])

      model.summary()

      delta = 1/number_of_samples
      l2_norm_clip = cfg['centralized_learning']['dpSGD']['l2_norm_clip']
      noise_multiplier = cfg['centralized_learning']['dpSGD']['noise_multiplier']
      num_microbatches = cfg['centralized_learning']['dpSGD']['num_microbatches']
      learning_rate = cfg['centralized_learning']['dpSGD']['learning_rate']
      used_microbatching = cfg['centralized_learning']['privacy_accountant']['used_microbatching']

      # Select your differentially private optimizer
      if cfg['centralized_learning']['training']['optimizer'] == 'DPSGD':
        optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate)

      elif cfg['centralized_learning']['training']['optimizer'] == 'SGD':
        optimizer = 'sgd'

      ################################################################################################

      model.compile(loss=cfg['centralized_learning']['training']['loss_function'], 
                    optimizer=optimizer, 
                    metrics=['accuracy'])

      # fit network
      model.fit(train_gen, 
                validation_data = test_gen, 
                steps_per_epoch = len(train_gen)//batch_size, 
                verbose=verbose, 
                epochs=epochs, 
                callbacks = [csv_logger, model_checkpoint_callback])

      ################################################################################################

      # push the experiment artifacts to S3

      
      #push this csv to S3 bucket
      
      s3_write_status = AWSwriter.push_data_to_s3(f'{myuuid}_centralizedLearning_csv.csv', 
                                        cfg['AWS']['trainingCSV']['bucket_name'], 
                                        f"{csv_folder_path}{myuuid}_centralizedLearning.csv")
      if s3_write_status:
          os.remove(f"{csv_folder_path}{myuuid}_centralizedLearning.csv")

       
      s3_model_write_status = AWSwriter.push_data_to_s3(f'{myuuid}_centralizedLearning_model.h5', 
                                        cfg['AWS']['models']['bucket_name'], 
                                        f"{model_checkpoint_folder_path}{myuuid}_centralizedLearning.h5")

      if s3_model_write_status:
          os.remove(f"{model_checkpoint_folder_path}{myuuid}_centralizedLearning.h5")

      ################################################################################################
      



if __name__ == "__main__":

  initialize(version_base= "1.1", config_path="./conf", job_name="decentralized_client_app")
  cfg = compose(config_name="flConfig")

  np.random.seed(cfg['centralized_learning']['data_distribution']['seed'])
  train_data = []
  test_data = []
  counter = 0
  client_set = set()

  broker = cfg['centralized_learning']['mqtt_broker']['ip']
  client = mqtt.Client(client_id= cfg['centralized_learning']['mqtt_broker']['server_id'])
  client.connect(broker,
                cfg['centralized_learning']['mqtt_broker']['port'], 
                cfg['centralized_learning']['mqtt_broker']['timeout'])


  print("Ready...")
  client.on_connect = on_connect
  client.on_message = on_message

  #client.loop_forever()
  
  client.loop_start()

  time.sleep(1000) # wait 1s

      
  client.loop_stop()
  print("loop stopped") 





