from models.BiLSTM_Attention import BiLSTM_with_Attention, BiLSTM_with_Attention_WE
from models.BiLSTM_CNN_Attention_Glove import LSTM_CNN_Attention_Glove
from models.bert import BERT
from models.LSTM import LSTM_Model, LSTM_Model_WE
from models.LSTM_Attention import LSTM_with_Attention, LSTM_with_Attention_WE
from models.LSTM_CNN import CNN_LSTM
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from argparse import Namespace
import os,time


#   Dataset : Ohsumed

#   Implementing All Models

if __name__ == "__main__":
    print(os.getpid())
    pid = os.getpid()
    file1 = open("tmp.txt", "w")
    file1.write(f"{pid}")
    file1.close()
    print("Waiting for execution...")

    time.sleep(60)
    train_df=pd.read_csv('Train.csv')
    train_df = train_df.sample(frac = 1)
    train_texts=train_df['Text'].values
    train_labels=train_df['Label'].values

    test_df=pd.read_csv('Test.csv')
    test_df = test_df.sample(frac = 1)
    test_texts=test_df['Text'].values
    test_labels=test_df['Label'].values

    #Model1
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"BiLSTM with Attention Start: {curr_time}\n")
    file1.close()
    print("Implementing BiLSTM with Attention Model")
    model = BiLSTM_with_Attention()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    model.save()
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"BiLSTM with Attention Stop: {curr_time}\n")
    file1.close()

    #Model2
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"BiLSTM with Attention and WE Start: {curr_time}\n")
    file1.close()
    print("Implementing BiLSTM with Attention and Word Embedding Model")
    model = BiLSTM_with_Attention_WE()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    model.save()
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"BiLSTM with Attention with WE Stop: {curr_time}\n")
    file1.close()

    #Model3
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"BiLSTM with CNN,Attention and Glove Word Embedding Start: {curr_time}\n")
    file1.close()
    print("Implementing BiLSTM with CNN,Attention and Glove Word Embedding Model")
    model = LSTM_CNN_Attention_Glove()
    model.fit(train_texts, train_labels)
    accuracy = model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    #model.save()
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"BiLSTM with CNN,Attention and Glove Word Embedding Stop: {curr_time}\n")
    file1.close()

    #Model4
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"BERT model by truncating maximum length of sentence to 512 Start: {curr_time}\n")
    file1.close()
    print("Implementing BERT model by truncating maximum length of sentence to 512")
    model = BERT(512, 8, 10, 2e-5)
    train_dataset_loader = model.preprocess(train_texts, train_labels)
    test_dataset_loader = model.preprocess(test_texts, test_labels)
    model.fit(train_dataset_loader)
    predicted_label = model.predict(test_dataset_loader)
    accuracy = accuracy_score(np.asarray(test_labels), np.asarray(predicted_label))
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"BERT model by truncating maximum length of sentence to 512 Stop: {curr_time}\n")
    file1.close()

    #Model5
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"BERT model by splitting one sentence into multiple sentences to maintain maximum length of 512 Start: {curr_time}\n")
    file1.close()
    print("Implementing BERT model by splitting one sentence into multiple sentences to maintain maximum length of 512")
    model = BERT(512, 8, 10, 2e-5)
    train_dataset_loader = model.preprocess_with_chunking(train_texts, train_labels)
    test_dataset_loader = model.preprocess_with_chunking(test_texts, test_labels)
    model.fit(train_dataset_loader)
    predicted_label = model.predict(test_dataset_loader)
    accuracy = accuracy_score(np.asarray(test_labels), np.asarray(predicted_label))
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    file1 = open("main_logs.txt", "a")  # append mode
    file1.write(f"BERT model by splitting one sentence into multiple sentences to maintain maximum length of 512 Stop: {curr_time}\n")
    file1.close()

    #Model6
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

    #Model7
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

    #Model8
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
    
    #Model9
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

    #Model10
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