import os
import re
import argparse
import pickle

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


def predict_dir(data_dir, model_dir, embeddings):

  for corpus_file in os.listdir(data_dir):
    if 'dev' not in corpus_file:
      continue
    print('Predicting:', corpus_file)

    corpus_path = os.path.join(data_dir, corpus_file)
    preds = predict(corpus_path, model_dir, embeddings)
    preds_file = corpus_file.split('.')[0] + '.npy'
    np.save(os.path.join(model_dir, preds_file), preds)


def predict(corpus_path, model_dir, embeddings):

  with open('word_index.pickle', 'rb') as fin:
    word_index = pickle.load(fin)
  with open('embedding_matrix.pickle', 'rb') as fin:
    embedding_matrix = pickle.load(fin)

  tweets, labels = get_data(corpus_path)

  tokenizer = Tokenizer()
  tokenizer.word_index = word_index
  sequences = tokenizer.texts_to_sequences(tweets)
  sequences = pad_sequences(sequences, maxlen=50)

  adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-9, decay=0.0, amsgrad=False)
  model = get_model(model_dir, corpus_path, embedding_matrix)
  corpus_file = os.path.basename(corpus_path)

  weights_paths = os.listdir(model_dir)
  search = re.sub('2018-', '', corpus_file)
  search = re.sub('.txt', '', search)
  search = re.sub('-dev', '', search)
  #print(search)
  weights_file = [match for match in weights_paths if search in match][0]
  weights_path = os.path.join(model_dir, weights_file)
  #weights_path = os.path.join(model_dir, corpus_file.split('.')[0] + '.hdf5')
  model.load_weights(weights_path)
  model.compile(loss='mean_squared_error', optimizer=adam)

  predictions = model.predict(sequences)
  #print(pearsonr(labels.reshape(-1, 1), predictions))
  return predictions


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a deep NLP model')
  parser.add_argument('-c', '--corpus_path',
                      help='Training corpus - txt file.')
  parser.add_argument('-d', '--data_dir',
                      help='Data directory. (for generating all predictions)')
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

  if args['corpus_path'] is not None: # predict file
    args.pop('data_dir')
    predict(**args)
  elif args['data_dir'] is not None: # predict directory
    args.pop('corpus_path')
    predict_dir(**args)
