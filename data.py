import os
import argparse
import pickle

import numpy as np
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm, trange
#from keras.preprocessing.text import pad_sequences


def prepare_word_index(data_dir):
  print('Preparing word index..')
  texts = []
  for filename in tqdm(os.listdir(data_dir)):
    if not filename.endswith('.txt'):
      continue
    filepath = os.path.join(data_dir, filename)

    with open(filepath, 'r') as fin:
      for line in fin:
        texts.append(line.strip().split('\t')[1])

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(texts)

  with open('word_index.pickle', 'wb') as fout:
    pickle.dump(tokenizer.word_index, fout)

  return tokenizer.word_index


def prepare_embedding_index(embedding_file):
  print('Preparing embedding index..')

  with open(embedding_file, 'r') as fin:
    # count lines for tqdm progress bar
    total = 0
    for line in fin:
      total += 1

  embedding_index = {}
  with tqdm(total=total) as pbar:
    with open(embedding_file, 'r') as fin:
      for line in fin:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector
        pbar.update(1)

  with open('embedding_index.pickle', 'wb') as fout:
    pickle.dump(embedding_index, fout)

  return embedding_index


def prepare_embedding_matrix(word_index, embedding_index):
  print('Preparing embedding matrix..')
  emb_dim = len(list(embedding_index.values())[0])
  embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
  for word, index in tqdm(word_index.items()):
    embedding = embedding_index.get(word)
    if embedding is not None:
      embedding_matrix[index] = embedding

  with open('embedding_matrix.pickle', 'wb') as fout:
    pickle.dump(embedding_matrix, fout)

  return embedding_matrix


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data_dir', required=True,
                      help='Data directory containing .txt files.')
  parser.add_argument('-e', '--embedding_file', required=True,
                      help='Path to embedding file.')

  args = vars(parser.parse_args())

  word_index = prepare_word_index(args['data_dir'])
  emb_index = prepare_embedding_index(args['embedding_file'])
  emb_matrix = prepare_embedding_matrix(word_index, emb_index)
