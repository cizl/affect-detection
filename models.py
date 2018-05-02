from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Conv1D, Dense, Flatten, LSTM, Bidirectional
from keras.layers import Dropout, BatchNormalization, GlobalMaxPool1D


def get_model(model_dir, corpus, embedding_matrix):
  outputs = None
  if '-reg-' in corpus:
    outputs = 1
  if '-oc-' in corpus:
    outputs = 3
  if '-c-' in corpus:
    outputs = 11
  print(outputs)

  if 'cnn' in model_dir:
    return get_cnn(embedding_matrix, outputs)
  if 'lstm' in model_dir:
    return get_lstm(embedding_matrix, outputs)
  else:
    print('Unknown model:',
          'make sure that model_dir contains model name. (cnn, lstm...)')

def create_embedding_layer(embedding_matrix):
  return Embedding(embedding_matrix.shape[0],
                   embedding_matrix.shape[1],
                   weights=[embedding_matrix],
                   input_length=50,
                   trainable=False)


def get_cnn(embedding_matrix, outputs):
  model = Sequential()
  model.add(create_embedding_layer(embedding_matrix))

#  model.add(Conv1D(128, 5, activation='relu')
#  model.add(BatchNormalization())
#  model.add(Conv1D(64, 5, activation='relu'))
#  model.add(BatchNormalization())
  model.add(Conv1D(100, 3, activation='relu'))
  model.add(BatchNormalization())
  #model.add(GlobalMaxPool1D())
  model.add(Flatten())
  model.add(Dense(outputs, activation="sigmoid"))
  print(model.summary())

  return model
  
  
def get_lstm(embedding_matrix, outputs, lstm_dim=50, dropout_rate=0.3, 
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

  model.add(Dense(outputs, activation="sigmoid"))

  return model 


