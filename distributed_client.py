# import libraries
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

from functools import reduce
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy_statement
import tensorflow_privacy
import re

from time import sleep
import random
import paho.mqtt.client as mqtt
import json
import sys

try:
    import cPickle as pickle
except:
    import pickle



def create_train_window_sequences(train_df, label_encoder, FEATURE_COLUMNS, window_size=100, number_of_samples=100):
  sequences = []
  for group_id, group in train_df.groupby("groups"):
    sequence_features = group[FEATURE_COLUMNS].values
    label = label_encoder.transform([np.unique(group['Class'].values)]).toarray()[0]
    sequences.append((sequence_features, label))

  # generate train data
  train_index_list = []
  train_window_sequences = []
  for _ in range(number_of_samples):
    random_sequence_index = np.random.randint(len(sequences))
    sequence, label = sequences[random_sequence_index]
    # random windows index
    start_index = np.random.randint(len(sequence) - window_size)
    train_index_list.append(start_index)
    features = sequence[start_index:start_index+window_size]
    train_window_sequences.append((features, label))
  return train_window_sequences



def create_test_window_sequences(test_df, label_encoder,FEATURE_COLUMNS,  window_size=100, number_of_samples=100):
  sequences = []
  for group_id, group in test_df.groupby("groups"):
    sequence_features = group[FEATURE_COLUMNS].values
    label = label_encoder.transform([np.unique(group['Class'].values)]).toarray()[0]
    sequences.append((sequence_features, label))

  # generate test data
  test_index_list = []
  test_window_sequences = []
  for _ in range(number_of_samples):
    random_sequence_index = np.random.randint(len(sequences))
    sequence, label = sequences[random_sequence_index]
    # random windows index
    start_index = np.random.randint(len(sequence) - window_size)
    test_index_list.append(start_index)
    features = sequence[start_index:start_index+window_size]
    test_window_sequences.append((features, label))
  return test_window_sequences


###########################################################################################################################################

def main(client_id, cfg):

  df = pd.read_csv(cfg['data_source']['path'])
  filtered_columns = ['Long_Term_Fuel_Trim_Bank1', 'Intake_air_pressure', 'Accelerator_Pedal_value', 'Fuel_consumption',
  'Torque_of_friction', 'Maximum_indicated_engine_torque', 'Engine_torque', 'Calculated_LOAD_value', 'Activation_of_Air_compressor',
  'Engine_coolant_temperature', 'Wheel_velocity_front_left-hand', 'Wheel_velocity_front_right-hand', 'Wheel_velocity_rear_left-hand',
  'Torque_converter_speed', 'Time(s)', 'Class']

  df[filtered_columns].head()

  count = 1
  groups = []

  for idx, row in enumerate(df[['Time(s)', 'Class']].iterrows()):
    if row[1][0] == 1:
      count += 1
      if idx == 0:
          count = 1
      groups.append(f'{row[1][1]}{count}')
    else:
      groups.append(f'{row[1][1]}{count}')

  df['groups'] = groups

  ###########################################################################################################################################

  data_distribution = cfg['centralized_learning']['data_distribution']

  ###########################################################################################################################################

  FEATURE_COLUMNS =  filtered_columns[:-2]

  test_df_list = []
  train_df_list = []
  for class_group in data_distribution[client_id]['training_record']:
    tmp_df = df[df['groups']==class_group]
    test_df_list.append(tmp_df.iloc[-int(len(tmp_df) * .3):, :])
    train_df_list.append(tmp_df.iloc[:int(len(tmp_df) * .7), :])

  train_df = pd.concat(train_df_list)
  test_df = pd.concat(test_df_list)

  label_encoder = OneHotEncoder()
  label_encoder.fit_transform(np.reshape(pd.concat([train_df, test_df]).Class.values, (-1, 1)))

  broker = cfg['centralized_learning']['mqtt_broker']['ip']
  client = mqtt.Client(client_id=client_id)
  res = client.connect(broker, 
                 cfg['centralized_learning']['mqtt_broker']['port'], 
                 cfg['centralized_learning']['mqtt_broker']['timeout'])
  print(res)

  train_window_sequences = create_train_window_sequences(train_df, 
                                                        label_encoder,
                                                        FEATURE_COLUMNS,
                                                        number_of_samples=cfg['centralized_learning']['data']['number_of_samples'])

  test_window_sequences = create_test_window_sequences(test_df, 
                                                      label_encoder, 
                                                      FEATURE_COLUMNS,
                                                      number_of_samples=cfg['centralized_learning']['data']['number_of_samples'])

  ###########################################################################################################################################

  data = {
    "client_id": client_id, 
    "train_window_sequences": train_window_sequences, 
    "test_window_sequences": test_window_sequences
  }

  # Send a few messages
  response = client.publish(cfg['centralized_learning']['mqtt_broker']['topic'], pickle.dumps(data), qos=1)
  print(response)



if __name__ == '__main__':

  initialize(version_base= "1.1", config_path="./conf", job_name="decentralized_client_app")
  cfg = compose(config_name="flConfig")

  client_id = sys.argv[1]
  np.random.seed(cfg['centralized_learning']['data_distribution']['seed'])
  main(client_id, cfg)

