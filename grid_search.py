import sys
import os
import itertools
import argparse
import pickle

import bpdb
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Conv1D, Dense, Flatten, LSTM, Bidirectional
from keras.layers import Dropout, BatchNormalization, GlobalMaxPool1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from metrics import PearsonCallback
from data import prepare_word_index, get_data
from models import create_embedding_layer
from models import architectures, param_grid


def model_parametrizer(architecture, param_grid, embedding_matrix):
  args = []
  for layer in architecture:
    args.append(list(itertools.product(*param_grid[layer])))

  all_args = list(itertools.product(*args))
  for args in tqdm(all_args):
    model = Sequential()
    model.add(create_embedding_layer(embedding_matrix))
    for layer, arg in zip(architecture, args):
      model.add(layer(*arg))

    model.add(Flatten())
#    print(model.summary())
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
        checkpoint = ModelCheckpoint(filepath=os.path.join(model_dir, 'weights.hdf5'), verbose=0, save_best_only=True)
        callbacks = [early_stopping, checkpoint]
#        if '-reg-' in corpus:
#          pearson = PearsonCallback()
#          callbacks.append(pearson)

        metrics = []
        if '-oc-' in corpus:
          metrics.append('acc')

        model.compile(loss=loss, optimizer='adam', metrics=metrics)
        model.fit(x_train, y_train, 
                  validation_data=(x_val, y_val),
                  epochs=300, 
                  batch_size=64,
                  callbacks=callbacks,
                  verbose=0)

def train(corpus_file, model_dir, embeddings, fresh_run=False, data_dir=None,
          epochs=3, batch_size=64, val_split=0.2):

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
#    fresh_run=True

  if fresh_run:
    word_index = prepare_word_index(data_dir)
    embedding_index = prepare_embedding_index(embeddings)
    embedding_matrix = prepare_embedding_matrix(word_index, embedding_index)
  else:
    with open('word_index.pickle', 'rb') as fin:
      word_index = pickle.load(fin)
    with open('embedding_index.pickle', 'rb') as fin:
      embedding_index = pickle.load(fin)
    with open('embedding_matrix.pickle', 'rb') as fin:
      embedding_matrix = pickle.load(fin)

  tweets, labels = get_data(corpus_file)

  tokenizer = Tokenizer()
  tokenizer.word_index = word_index
  sequences = tokenizer.texts_to_sequences(tweets)
  sequences = pad_sequences(sequences, maxlen=50)

  x_train, x_val, y_train, y_val = train_test_split(
      sequences, labels, test_size=val_split, random_state=42)
  print(x_train[0], y_train[0])
  print(y_train.shape)

  early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
  checkpoint = ModelCheckpoint(filepath=os.path.join(model_dir, 'weights.hdf5'), verbose=1, save_best_only=True)
  callbacks = [early_stopping, checkpoint]

  if '-reg-' in corpus_file:
    pearson = PearsonCallback()
    callbacks.append(pearson)

  metrics = []
  if '-oc-' in corpus_file:
    metrics.append('acc')

  if '-reg-' in corpus_file:
    loss = 'mean_squared_error'
  else:
    loss = 'categorical_crossentropy'

  grid_search(x_train, y_train, architectures, param_grid, 5, '-reg-', embedding_matrix, model_dir)
 


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a deep NLP model')
  parser.add_argument('-c', '--corpus_file', required=True,
                      help='Training corpus - txt file.')
  parser.add_argument('-m', '--model_dir', required=True,
                      help='Model data dir. (weights, tokens, etc.)')
  parser.add_argument('-em', '--embeddings', required=True,
                      help='Embeddings file.')
  parser.add_argument('-e', '--epochs', type=int, default=3,
                      help='Number of training epochs.')
  parser.add_argument('-b', '--batch_size', type=int, default=64,
                      help='Size of training minibatch.')
  parser.add_argument('-v', '--val_split', type=float, default=0.2,
                      help='Percentage of validation data.')
  parser.add_argument('-f', '--fresh_run', action='store_true', 
                      default=False,
                      help='Do a completely new run')
  parser.add_argument('-d', '--data_dir', type=str, default=None,
                      help='Data directory, contains all corpora. '
                      + 'Used in fresh run')
  parser.add_argument('-g', '--gpu', type=str,
                      help='Which GPU to use (CUDA_VISIBLE_DEVICES).')

  args = vars(parser.parse_args())

  gpu = args.pop('gpu')
  if gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

  train(**args)
