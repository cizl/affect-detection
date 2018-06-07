import sys
import os
import itertools
import bpdb

import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Conv1D, Dense, Flatten, LSTM, Bidirectional
from keras.layers import Dropout, BatchNormalization, GlobalMaxPool1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from metrics import PearsonCallback


architectures = [
  [Conv1D, Dropout],
#  [Conv1D, Dropout, Conv1D, Dropout]
]

param_grid = {
  Conv1D: [[10, 50, 100, 200, 400],
           [1, 2, 3, 5],
#           ['relu'],
           ],
#  Dropout: [[0, 0.2, 0.5, 0.8]],
  Dropout: [[0, 0.5, 0.8]],
}

def get_model(model_dir, corpus, embedding_matrix):
  """Model multiplexor."""

  model = None
  if 'cnn' in model_dir:
    print('cnn')
    model = get_cnn(embedding_matrix)
  elif 'lstm' in model_dir:
    print('lstm')
    model = get_lstm(embedding_matrix)
  else:
    print('Unknown model:',
          'make sure that model_dir contains model name. (cnn, lstm...)')
    sys.exit(-1)

  if '-reg-' in corpus:
    model.add(Dense(1, activation='sigmoid'))
  if '-oc-' in corpus:
    model.add(Dense(1, activation='linear'))
  if '-c-' in corpus:
    model.add(Dense(11, activation='sigmoid'))

#  print(model.summary())
  return model

def create_embedding_layer(embedding_matrix):
  return Embedding(embedding_matrix.shape[0],
                   embedding_matrix.shape[1],
                   weights=[embedding_matrix],
                   input_length=50,
                   trainable=False)


def get_cnn(embedding_matrix):
  model = Sequential()
  model.add(create_embedding_layer(embedding_matrix))
  model.add(Conv1D(200, 3, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.8))
  #model.add(GlobalMaxPool1D())
  model.add(Flatten())

  return model
  
  
def get_lstm(embedding_matrix, lstm_dim=50, dropout_rate=0.3, 
             batch_norm=True):

  model = Sequential()
  model.add(create_embedding_layer(embedding_matrix))
  model.add(Bidirectional(LSTM(lstm_dim, dropout=dropout_rate, 
                               activation='sigmoid', return_sequences=True)))
  model.add(GlobalMaxPool1D())
  if batch_norm:
    model.add(BatchNormalization())
  if embedding_matrix is not None:
    model.add(Dropout(dropout_rate))
  model.add(Dense(lstm_dim, activation="relu"))
  if batch_norm:
    model.add(BatchNormalization())

  return model 


