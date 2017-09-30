from keras.models import load_model

import numpy as np

model = load_model('LSTM2_stockmarket_5000_GM.h5py')

testArray = np.array([[0.14754], [0.16894], [0.185462], [0.15698], [0.15642], [0.112654], [0.103658], [0.169845], [0.1962545], [0.20468]])

predicted = [testArray]
predicted = np.array(predicted)
outputFinalArray = model.predict(predicted)