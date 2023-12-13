from LSTM import LSTM_Model, LSTM_Model_WE
from LSTM_Attention import LSTM_with_Attention, LSTM_with_Attention_WE
from LSTM_CNN import CNN_LSTM
import preprocessdata
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from argparse import Namespace
import os,time


#   Dataset : Ohsumed

#   Implementing All Models

if __name__ == "__main__":
    pid = os.getpid()
    file1 = open("tmp.txt", "w")
    file1.write(f"{pid}")
    file1.close()

    time.sleep(60)
    train_df=pd.read_csv('Train.csv')
    train_df = train_df.sample(frac = 1)
    train_texts=train_df['Text'].values
    train_labels=train_df['Label'].values

    test_df=pd.read_csv('Test.csv')
    test_df = test_df.sample(frac = 1)
    test_texts=test_df['Text'].values
    test_labels=test_df['Label'].values

    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"SimpleLSTM Start: {curr_time}\n")
    file1.close()
    print("Implementing Simple LSTM Model")
    model = LSTM_Model()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    model.save()
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"SimpleLSTM Stop: {curr_time}\n")
    file1.close()

    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"LSTM with WE Start: {curr_time}\n")
    file1.close()
    print("Implementing LSTM Model with Word Embedding")
    model = LSTM_Model_WE()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    model.save()
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"LSTM with WE Stop: {curr_time}\n")
    file1.close()

    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"LSTM with Att Start: {curr_time}\n")
    file1.close()
    print("Implementing LSTM with Attention Model")
    model = LSTM_with_Attention()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    model.save()
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"LSTM with Att Stop: {curr_time}\n")
    file1.close()
    
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"LSTM with Att WE Start: {curr_time}\n")
    file1.close()
    print("Implementing LSTM with Attention and Word Embedding Model")
    model = LSTM_with_Attention_WE()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    model.save()
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"LSTM with Att WE Stop: {curr_time}\n")
    file1.close()

    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"CNN LSTM Start: {curr_time}\n")
    file1.close()
    print("Implementing CNN LSTM Model")
    args=Namespace(dataset='ohsumed', dataset_id='lemmatized', num_classes=23, run_id='', embpath='', embsize=300, maxlen=1000, vocabsize=40000, static=False)
    new_model = CNN_LSTM(args)
    new_model.evaluate()
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"CNN LSTM Stop: {curr_time}\n")
    file1.close()