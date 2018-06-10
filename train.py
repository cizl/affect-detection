import os
import argparse
import pickle

import bpdb
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from data import prepare_word_index, get_data
from data import prepare_embedding_index, prepare_embedding_matrix
from data import prepare_affect_index, prepare_affect_matrix
from models import get_model
from metrics import PearsonCallback


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

  tokenizer = Tokenizer()
  tokenizer.word_index = word_index
  sequences = tokenizer.texts_to_sequences(tweets)
  sequences = pad_sequences(sequences, maxlen=50)

  x_train, x_val, y_train, y_val = train_test_split(
      sequences, labels, test_size=val_split, random_state=42)
  print(x_train[0], y_train[0])
  print(y_train.shape)

  early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')
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
  elif '-oc' in corpus_file:
    #loss = 'sparse_categorical_crossentropy'
    loss = 'mean_squared_error'
  elif '-c-' in corpus_file:
    loss = 'categorical_crossentropy'

  rmsprop = keras.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
  adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-9, decay=0.0, amsgrad=False)
  model = get_model(model_dir, corpus_file, embedding_matrix)
  model.compile(loss=loss, optimizer=adam, metrics=metrics)
  
  print(model.summary())
  model.fit(x_train, y_train, 
            validation_data=(x_val, y_val),
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=callbacks)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a deep NLP model')
  parser.add_argument('-c', '--corpus_file', required=True,
                      help='Training corpus - txt file.')
  parser.add_argument('-m', '--model_dir', required=True,
                      help='Model data dir. (weights, tokens, etc.)')
  parser.add_argument('-em', '--embeddings', required=True,
                      help='Embeddings file.')
  parser.add_argument('-e', '--epochs', type=int, default=300,
                      help='Number of training epochs.')
  parser.add_argument('-a', '--affect_lexicon', required=False,
                      help='Path to affect lexicon (txt).')
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
