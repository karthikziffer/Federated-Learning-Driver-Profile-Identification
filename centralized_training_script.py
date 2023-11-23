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
from IPython.core.debugger import set_trace
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import math
import io
import datetime as dt
import re
from math import sqrt
import pandas as pd

from functools import reduce
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy_statement
import tensorflow_privacy
import re

##########################################################################################################################################


DATA_PATH = "./Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv"


df = pd.read_csv(DATA_PATH)
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


##########################################################################################################################################


data_distribution = {

  "clients": ["client1", "client2", "client3"],
  "client1": {
        "training_record": ["D3", "B9", "D2", "B5", "A1", "A42", "E20", "E34"],
        "testing_record":  ["D3", "B9"]
        },
  "client2": {
        "training_record": ["D36", "B7", "D35", "B40", "A10", "A42", "E21", "E34"],
        "testing_record":  ["D35", "B40"]
        },
  "client3": {
        "training_record": ["D44", "B37", "D4", "B33", "A41", "A42", "E32", "E34"],
        "testing_record":  ["D4", "B33"]
        }
  }


##########################################################################################################################################


filtered_columns = ['Long_Term_Fuel_Trim_Bank1', 'Intake_air_pressure', 'Accelerator_Pedal_value', 'Fuel_consumption',
'Torque_of_friction', 'Maximum_indicated_engine_torque', 'Engine_torque', 'Calculated_LOAD_value', 'Activation_of_Air_compressor',
'Engine_coolant_temperature', 'Wheel_velocity_front_left-hand', 'Wheel_velocity_front_right-hand', 'Wheel_velocity_rear_left-hand',
'Torque_converter_speed', 'Time(s)', 'Class']

FEATURE_COLUMNS =  filtered_columns[:-2]

test_df_list = []
train_df_list = []
for class_group in data_distribution["client1"]['training_record']:
  # print(class_group)
  tmp_df = df[df['groups']==class_group]
  # print(tmp_df.index)
  # print(tmp_df.loc[:int(len(tmp_df) * .7), :].index, "\n\n",  tmp_df.loc[-int(len(tmp_df) * .3):, :].index)
  # print(len(tmp_df))
  test_df_list.append(tmp_df.iloc[-int(len(tmp_df) * .3):, :])
  # print(len(tmp_df))
  train_df_list.append(tmp_df.iloc[:int(len(tmp_df) * .7), :])
  # print(len(tmp_df), len(tmp_df) * .3)

train_df = pd.concat(train_df_list)
test_df = pd.concat(test_df_list)


label_encoder = OneHotEncoder()
label_encoder.fit_transform(np.reshape(pd.concat([train_df, test_df]).Class.values, (-1, 1)))

##########################################################################################################################################


def create_train_window_sequences(train_df, label_encoder, window_size=100, number_of_samples=100):
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



def create_test_window_sequences(test_df, label_encoder, window_size=100, number_of_samples=100):
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


##########################################################################################################################################




class DriverDatasetGenerator(keras.utils.Sequence):

  def __init__(self, sequences, batch_size, n_channels, window_size, classes, shuffle=True):
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
      print(list_IDs_temp)
      X, y = self.__data_generation(list_IDs_temp)
      return X, y

  def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

      X = np.empty((self.batch_size, self.window_size, self.n_channels))
      y = np.empty((self.batch_size, self.n_classes))

      # Generate data
      for i, _ in enumerate(list_IDs_temp):
          # Store sample
          print(self.sequences[_])
          X[i,], y[i,] = self.sequences[_]
      return X, y


##########################################################################################################################################


window_size = 100
number_of_samples = 1000
verbose, epochs = 1, 1000
n_timesteps, n_features, n_outputs, batch_size = window_size, 14, 4, 32


train_window_sequences = create_train_window_sequences(train_df, label_encoder, window_size, number_of_samples)
test_window_sequences = create_test_window_sequences(test_df, label_encoder, window_size, number_of_samples)

print(train_window_sequences[0], type(train_window_sequences))

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

# model = Sequential()
# model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
# model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
# model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(n_outputs, activation='softmax'))


# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv1D(filters=32, kernel_size=5,
#                       strides=1,
#                       activation="relu",
#                       input_shape=(n_timesteps,n_features)),
#   tf.keras.layers.LSTM(64, return_sequences =True),
#   tf.keras.layers.LSTM(64, return_sequences =True),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(30, activation="relu"),
#   tf.keras.layers.Dense(10, activation="softmax")
# ])


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

################################################################################################

model.summary()


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
        print(f"\n Privacy Budget: {privacy_budget}  {self.tmp_epoch}  {epoch}")

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
        print(f"\n Gradient norm: {gradient_norm}")


################################################################################################

delta = 1/number_of_samples
l2_norm_clip = 20
noise_multiplier = 0.1
num_microbatches = 1
learning_rate = 0.001
used_microbatching = False

# Select your differentially private optimizer
dp_optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate)

# dp_optimizer = 'sgd'

################################################################################################

model.compile(loss='categorical_crossentropy', optimizer=dp_optimizer, metrics=['accuracy'])

# fit network
model.fit(train_gen, validation_data = test_gen, steps_per_epoch = len(train_gen)//batch_size, verbose=verbose, epochs=epochs)

