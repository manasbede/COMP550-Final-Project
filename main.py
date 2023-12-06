from LSTM import LSTM_Model, LSTM_Model_WE
from LSTM_Attention import LSTM_with_Attention, LSTM_with_Attention_WE
from LSTM_CNN import CNN_LSTM
import preprocessdata
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from argparse import Namespace


#   Dataset : Ohsumed

#   Implementing All Models

if __name__ == "__main__":
    train_df=pd.read_csv('Train.csv')
    train_df = train_df.sample(frac = 1)
    train_texts=train_df['Text'].values
    train_labels=train_df['Label'].values

    test_df=pd.read_csv('Test.csv')
    test_df = test_df.sample(frac = 1)
    test_texts=test_df['Text'].values
    test_labels=test_df['Label'].values

    print("Implementing Simple LSTM Model")
    model = LSTM_Model()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    print("Implementing LSTM Model with Word Embedding")
    model = LSTM_Model_WE()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    print("Implementing LSTM with Attention Model")
    model = LSTM_with_Attention()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    print("Implementing LSTM with Attention and Word Embedding Model")
    model = LSTM_with_Attention_WE()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    print("Implementing CNN LSTM Model")
    args=Namespace(dataset='ohsumed', dataset_id='lemmatized', num_classes=23, run_id='', embpath='', embsize=300, maxlen=1000, vocabsize=40000, static=False)
    new_model = CNN_LSTM(args)
    new_model.evaluate()