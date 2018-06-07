import numpy as np
from keras.callbacks import Callback
from scipy.stats import pearsonr


class PearsonCallback(Callback):

  def on_epoch_end(self, epochs, logs={}):
    y_true = self.validation_data[1]
    y_pred = (np.asarray(self.model.predict(self.validation_data[0])))
#    for a, b in zip(y_true, y_pred):
#      print(a, b)
    
    print('\tPearson R:', pearsonr(y_true, y_pred))
