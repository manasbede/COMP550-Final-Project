from LSTM import LSTM_Model
import preprocessdata
from sklearn.metrics import accuracy_score
import numpy as np


#   Dataset : Ohsumed
train_texts, train_labels = preprocessdata.read_data_from_folders('training')
test_texts, test_labels = preprocessdata.read_data_from_folders('test')

#   Apply data on LSTM
model=LSTM_Model()
model.fit(train_texts,train_labels)
prediction=model.predict(test_texts)
accuracy = accuracy_score(np.asarray(test_labels), prediction)
print(accuracy)