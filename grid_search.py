import sys
import os
import itertools
import argparse
import pickle

import bpdb
import numpy as np
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Conv1D, Dense, Flatten, LSTM, Bidirectional
from keras.layers import Dropout, BatchNormalization, GlobalMaxPool1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tqdm import tqdm

from metrics import PearsonCallback
from data import prepare_word_index, get_data
from data import prepare_embedding_index, prepare_embedding_matrix
from data import prepare_affect_index, prepare_affect_matrix
from models import create_embedding_layer
from models import architectures, param_grid


def build_model(architecture, params, embedding_matrix, verbose=False):
  model = Sequential()
  model.add(create_embedding_layer(embedding_matrix))
  for layer, arg in zip(architecture, params):
    model.add(layer(*arg))
 
  model.add(Flatten())
  if verbose:
    print(model.summary())
  
  return model

def parametrizer(architecture, param_grid):
  args = []
  for layer in architecture:
    args.append(list(itertools.product(*param_grid[layer])))

  all_args = list(itertools.product(*args))
  for args in tqdm(all_args):
    yield args

def grid_search(X, y, val_x, val_y, architecture_grid, param_grid, n_folds, corpus, embedding_matrix, model_dir):
  best_score = 0 
  scores = {}
  models = {}

  for architecture in architecture_grid:
    param_gen = parametrizer(architecture, param_grid)

    for params in param_gen:
      print('params:', params)
      avg_score = 0
      for l in range(n_folds):
#      for k in range(1):
        val_start = l * len(X) // n_folds
        val_end = (l+1) * len(X) // n_folds
        x_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        x_train = np.vstack([X[:val_start], X[val_end:]])
        y_train = np.concatenate([y[:val_start], y[val_end:]])

        model = build_model(architecture, params, embedding_matrix)
#        print(model.summary())
        if '-reg-' in corpus:
          model.add(Dense(1, activation='linear'))
        if '-oc-' in corpus:
          model.add(Dense(1, activation='linear'))
        if '-c-' in corpus:
          model.add(Dense(11, activation='sigmoid'))

        if '-reg-' in corpus:
          loss = 'mean_squared_error'
        elif '-oc' in corpus:
          loss = 'sparse_categorical_crossentropy'
        elif '-c-' in corpus:
          loss = 'categorical_crossentropy'

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(filepath=os.path.join(model_dir, 'weights.hdf5'), verbose=0, save_best_only=True)
        callbacks = [early_stopping, checkpoint]#, PearsonCallback()]

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

        n_params = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        y_pred = model.predict(val_x).reshape(1, -1)[0]
        score = pearsonr(val_y, y_pred)[0]
        print('score:', score)
        avg_score += score

      avg_score /= n_folds
      print('avg:', avg_score, params)

      try:
        if avg_score > scores[n_params]:
          scores[n_params] = avg_score
          models[n_params] = (architecture, params)
      except KeyError:
        scores[n_params] = avg_score
        models[n_params] = (architecture, params)
      
      with open('history.pickle', 'wb') as fout:
        pickle.dump((scores, models), fout)

  return scores, models


def train(corpus_file, model_dir, embeddings, affect_lexicon, fresh_run=False, data_dir=None,
          epochs=3, batch_size=64, val_split=0.2):

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  if fresh_run:
    word_index = prepare_word_index(data_dir)
    embedding_index = prepare_embedding_index(embeddings)
    embedding_matrix = prepare_embedding_matrix(word_index, embedding_index)
    affect_index = prepare_affect_index(affect_lexicon)
    affect_matrix = prepare_affect_matrix(word_index, affect_index)
  else:
    with open('word_index.pickle', 'rb') as fin:
      word_index = pickle.load(fin)
    with open('embedding_index.pickle', 'rb') as fin:
      embedding_index = pickle.load(fin)
    with open('embedding_matrix.pickle', 'rb') as fin:
      embedding_matrix = pickle.load(fin)
    with open('affect_index.pickle', 'rb') as fin:
      affect_index = pickle.load(fin)
    with open('affect_matrix.pickle', 'rb') as fin:
      affect_matrix = pickle.load(fin)

  tweets, labels = get_data(corpus_file)
  print(tweets[:5])
  print(labels[:5])

  tokenizer = Tokenizer()
  tokenizer.word_index = word_index
  sequences = tokenizer.texts_to_sequences(tweets)
  sequences = pad_sequences(sequences, maxlen=50)

  x_train, x_val, y_train, y_val = train_test_split(
      sequences, labels, test_size=val_split, random_state=42)

  scores, models = grid_search(x_train, y_train, x_val, y_val, architectures, param_grid, 5, corpus_file, embedding_matrix, model_dir)
 
  with open('history.pickle', 'wb') as fout:
    pickle.dump((scores, models), fout)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a deep NLP model')
  parser.add_argument('-c', '--corpus_file', required=True,
                      help='Training corpus - txt file.')
  parser.add_argument('-m', '--model_dir', required=True,
                      help='Model data dir. (weights, tokens, etc.)')
  parser.add_argument('-em', '--embeddings', required=True,
                      help='Embeddings file.')
  parser.add_argument('-a', '--affect_lexicon', required=True,
                      help='Path to affect lexicon (txt).')
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
