import os
import argparse
import pickle
import re

import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from data import prepare_word_index, get_data
from data import prepare_embedding_index, prepare_embedding_matrix
from models import get_model
from metrics import PearsonCallback


def predict(corpus_file, model_dir, embeddings):

  with open('word_index.pickle', 'rb') as fin:
    word_index = pickle.load(fin)
  with open('embedding_index.pickle', 'rb') as fin:
    embedding_index = pickle.load(fin)
  with open('embedding_matrix.pickle', 'rb') as fin:
    embedding_matrix = pickle.load(fin)

  keys, tweets, labels = get_data(corpus_file, return_keys=True)

  tokenizer = Tokenizer()
  tokenizer.word_index = word_index
  sequences = tokenizer.texts_to_sequences(tweets)
  sequences = pad_sequences(sequences, maxlen=50)

  adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-9, decay=0.0, amsgrad=False)
  model = get_model(model_dir, corpus_file, embedding_matrix)
  model.load_weights(os.path.join(model_dir, 'weights.hdf5'))
  model.compile(loss='mean_squared_error', optimizer=adam)

  predictions = model.predict(sequences)

  os.makedirs('submissions', exist_ok=True)
  submission = os.path.basename(corpus_file)
  submission = re.sub('2018-', '', submission)
  submission = re.sub('-En-', '_en_', submission)
  submission = re.sub('-test', '_pred', submission)
  with open(os.path.join('submissions', submission), 'w') as fout:
    print('ID\tTweet\tAffect\tDimension\tIntensity\tScore', file=fout)
    for k, t, l, p in zip(keys, tweets, labels, predictions):
      print('\t'.join([k, t, l, str(p[0])]), file=fout)
    
  print(pearsonr(labels.reshape(-1, 1), predictions))
  return predictions


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a deep NLP model')
  parser.add_argument('-c', '--corpus_file', required=True,
                      help='Training corpus - txt file.')
  parser.add_argument('-m', '--model_dir', required=True,
                      help='Model data dir. (weights, tokens, etc.)')
  parser.add_argument('-em', '--embeddings', required=True,
                      help='Embeddings file.')
  parser.add_argument('-g', '--gpu', type=str,
                      help='Which GPU to use (CUDA_VISIBLE_DEVICES).')

  args = vars(parser.parse_args())

  gpu = args.pop('gpu')
  if gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

  predict(**args)
