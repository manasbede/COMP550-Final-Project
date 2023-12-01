import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


class LSTM_Model():
    def __init__(self):
        self.model = Sequential()
        self.tokenizer = Tokenizer(num_words=1000)
        self.max_sequence_length = 10  # maximum length of input sequences

    def fit(self,train_data, train_label):
        # Tokenize and preprocess the text data
        one_hot_labels = tf.keras.utils.to_categorical(train_label, num_classes=len(set(train_label)))
        self.tokenizer.fit_on_texts(train_data)
        sequences = self.tokenizer.texts_to_sequences(train_data)
        vocab_size = len(self.tokenizer.word_index) + 1

        train_label=np.asarray(train_label)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)


        # Define the LSTM model architecture
        self.model.add(Embedding(vocab_size, 16, input_length=self.max_sequence_length))
        self.model.add(LSTM(units=32))
        self.model.add(Dense(len(set(train_label)), activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        self.model.fit(padded_sequences, one_hot_labels, epochs=10, batch_size=32)

    def predict(self,test_data):
        new_sequences = self.tokenizer.texts_to_sequences(test_data)
        new_padded_sequences = pad_sequences(new_sequences, maxlen=self.max_sequence_length)
        predictions = self.model.predict(new_padded_sequences)
        return predictions