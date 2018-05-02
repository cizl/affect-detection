import os
import argparse
import pickle

import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from data import prepare_word_index
from data import prepare_embedding_index, prepare_embedding_matrix
from models import get_model


def train(corpus_file, model_dir, embeddings, fresh_run=False, data_dir=None,
          epochs=3, batch_size=64, val_split=0.2):

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
#    fresh_run=True

  if fresh_run:
    word_index = prepare_word_index(data_dir)
    embedding_index = prepare_embedding_index(embeddings)
    emb_matrix = prepare_embedding_matrix(word_index, embedding_index)
  else:
    with open('word_index.pickle', 'rb') as fin:
      word_index = pickle.load(fin)
    with open('embedding_index.pickle', 'rb') as fin:
      embedding_index = pickle.load(fin)
    with open('embedding_matrix.pickle', 'rb') as fin:
      embedding_matrix = pickle.load(fin)

  keys = []
  tweets = []
  labels = []
  with open(corpus_file, 'r') as fin:
    header = True
    for line in fin:
      if header:
        header = False
        continue
      key, tweet, *label = line.strip().split('\t')
      if '-oc-' in corpus_file:
        label = int(label[-1].split(':')[0])-1
      keys.append(key)
      tweets.append(tweet)
      labels.append(label)

  tokenizer = Tokenizer()
  tokenizer.word_index = word_index
  sequences = tokenizer.texts_to_sequences(tweets)
  sequences = pad_sequences(sequences, maxlen=50)

  print(labels[0])
  if '-c-' in corpus_file:
    labels = np.array([[float(x) for x in l] for l in labels])
  else:
    labels = to_categorical(labels)
  print(labels[0])

  x_train, x_val, y_train, y_val = train_test_split(
      sequences, labels, test_size=val_split, random_state=42)
  print(x_train.shape, y_train.shape)

  rmsprop = keras.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
  adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  model = get_model(model_dir, corpus_file, embedding_matrix)
  model.compile(loss='categorical_crossentropy',
                optimizer=rmsprop,
                metrics=['acc'])

  model.fit(x_train, y_train, validation_data=(x_val, y_val),
            epochs=epochs, batch_size=batch_size)


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
