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

from metrics import PearsonCallback


architectures = [
  [Conv1D, Dropout],
  [Conv1D, Dropout, Conv1D, Dropout]
]

param_grid = {
  Conv1D: [[10, 50, 100, 200, 400],
           [1, 2, 3, 4, 5, 6, 7],
#           ['relu'],
           ],
  Dropout: [[0, 0.1, 0.2, 0.4, 0.6, 0.8]],
}

def model_parametrizer(architecture, param_grid, embedding_matrix):
  args = []
  for layer in architecture:
    args.append(list(itertools.product(*param_grid[layer])))

  all_args = list(itertools.product(*args))
  for args in all_args:
    model = Sequential()
    model.add(create_embedding_layer(embedding_matrix))
    for layer, arg in zip(architecture, args):
      model.add(layer(*arg))

    model.add(Flatten())
    print(model.summary())
    yield model

def grid_search(X, y, architecture_grid, parameter_grid, n_folds, corpus, embedding_matrix, model_dir):
  for i in range(n_folds):
    val_start = i * len(X) // n_folds
    val_end = (i+1) * len(X) // n_folds
    x_val = X[val_start:val_end]
    y_val = y[val_start:val_end]
#    bpdb.set_trace()
    x_train = np.vstack([X[:val_start], X[val_end:]])
    y_train = np.concatenate([y[:val_start], y[val_end:]])

    for architecture in architecture_grid:
      model_gen = model_parametrizer(architecture, parameter_grid, embedding_matrix)
      for model in model_gen:
        if '-reg-' in corpus:
          model.add(Dense(1, activation='sigmoid'))
        if '-oc-' in corpus:
          model.add(Dense(1, activation='linear'))
        if '-c-' in corpus:
          model.add(Dense(11, activation='sigmoid'))

        if '-reg-' in corpus:
          loss = 'mean_squared_error'
        else:
          loss = 'categorical_crossentropy'

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(filepath=os.path.join(model_dir, 'weights.hdf5'), verbose=1, save_best_only=True)
        callbacks = [early_stopping, checkpoint]
        if '-reg-' in corpus:
          pearson = PearsonCallback()
          callbacks.append(pearson)

        metrics = []
        if '-oc-' in corpus:
          metrics.append('acc')

        model.compile(loss=loss, optimizer='adam', metrics=metrics)
        model.fit(x_train, y_train, 
                  validation_data=(x_val, y_val),
                  epochs=300, 
                  batch_size=64,
                  callbacks=callbacks)


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

  print(model.summary())
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


