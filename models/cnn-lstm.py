import nltk
from tensorflow.keras.layers import GlobalAveragePooling1D

nltk.download('punkt')
import pandas as pd
import tensorflow as tf
import keras
from keras_self_attention import SeqSelfAttention
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
import numpy as np
from keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class CNN_LSTM_Attention_Glove:
    def __init__(self):
        self.embedding_dim = 300
        self.lstm_out = 296
        self.max_words = 350
        self.tokenizer = Tokenizer(num_words=self.max_words, split=' ')
        self.model = Sequential()
        self.label_encoder = LabelEncoder()
        self.glove_path = '/content/drive/MyDrive/glove.6B.300d.txt'

    def load_glove_vectors(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            embeddings_index = {}
            for line in file:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def fit(self, train_texts, train_labels):
        # Tokenize text data
        path = self.glove_path
        glove_vectors = self.load_glove_vectors(path)

        self.tokenizer.fit_on_texts(train_texts)
        X_train = self.tokenizer.texts_to_sequences(train_texts)
        X_train = pad_sequences(X_train, maxlen=self.max_words)
        y_train = self.label_encoder.fit_transform(train_labels)

        num_words = len(self.tokenizer.word_index) + 1
        embedding_matrix = np.zeros((num_words, self.embedding_dim))

        for word, i in self.tokenizer.word_index.items():
            if word in glove_vectors:
                embedding_matrix[i] = glove_vectors[word]

        vocab_size = len(self.tokenizer.word_index) + 1

        embedding_layer = Embedding(vocab_size, self.embedding_dim, weights=[embedding_matrix],
                                    input_length=self.max_words,
                                    trainable=False)
        self.model.add(embedding_layer)
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(filters=128,
                              kernel_size=3,
                              padding='valid',
                              activation='relu',
                              strides=1))

        self.model.add(MaxPooling1D(pool_size=4))
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(filters=128,
                              kernel_size=3,
                              padding='valid',
                              activation='relu',
                              strides=1))

        self.model.add(MaxPooling1D(pool_size=4))

        # Add LSTM layer with return_sequences=True to get the sequence output
        self.model.add(Bidirectional(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

        self.model.add(Bidirectional(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

        # Add Attention layer
        self.model.add(SeqSelfAttention(attention_activation='sigmoid'))

        # Use GlobalAveragePooling1D to get a fixed-size output regardless of the sequence length
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(len(np.unique(train_labels)), activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 32
        epochs = 100
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def predict(self, test_data, test_labels):
        new_sequences = self.tokenizer.texts_to_sequences(test_data)
        new_padded_sequences = pad_sequences(new_sequences)
        y_test = self.label_encoder.fit_transform(test_labels)
        score, acc = self.model.evaluate(new_padded_sequences, y_test, verbose=2)
        return acc


class LSTM_CNN_Attention_Glove:
    def __init__(self):
        self.embedding_dim = 300
        self.lstm_out = 296
        self.max_words = 350
        self.tokenizer = Tokenizer(num_words=self.max_words, split=' ')
        self.model = Sequential()
        self.label_encoder = LabelEncoder()
        self.glove_path = '/content/drive/MyDrive/glove.6B.300d.txt'

    def load_glove_vectors(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            embeddings_index = {}
            for line in file:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def fit(self, train_texts, train_labels):
        # Tokenize text data
        path = self.glove_path
        glove_vectors = self.load_glove_vectors(path)

        self.tokenizer.fit_on_texts(train_texts)
        X_train = self.tokenizer.texts_to_sequences(train_texts)
        X_train = pad_sequences(X_train, maxlen=self.max_words)
        y_train = self.label_encoder.fit_transform(train_labels)

        num_words = len(self.tokenizer.word_index) + 1
        embedding_matrix = np.zeros((num_words, self.embedding_dim))

        for word, i in self.tokenizer.word_index.items():
            if word in glove_vectors:
                embedding_matrix[i] = glove_vectors[word]

        vocab_size = len(self.tokenizer.word_index) + 1

        embedding_layer = Embedding(vocab_size, self.embedding_dim, weights=[embedding_matrix],
                                    input_length=self.max_words,
                                    trainable=False)

        self.model.add(embedding_layer)
        self.model.add(Dropout(0.5))

        # Add LSTM layer with return_sequences=True to get the sequence output
        self.model.add(Bidirectional(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

        self.model.add(Bidirectional(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

        self.model.add(Conv1D(filters=128,
                              kernel_size=3,
                              padding='valid',
                              activation='relu',
                              strides=1))

        self.model.add(MaxPooling1D(pool_size=4))
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(filters=128,
                              kernel_size=3,
                              padding='valid',
                              activation='relu',
                              strides=1))

        self.model.add(MaxPooling1D(pool_size=4))

        # Add Attention layer
        self.model.add(SeqSelfAttention(attention_activation='sigmoid'))

        # Use GlobalAveragePooling1D to get a fixed-size output regardless of the sequence length
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(len(np.unique(train_labels)), activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 32
        epochs = 25
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def predict(self, test_data, test_labels):
        new_sequences = self.tokenizer.texts_to_sequences(test_data)
        new_padded_sequences = pad_sequences(new_sequences)
        y_test = self.label_encoder.fit_transform(test_labels)
        score, acc = self.model.evaluate(new_padded_sequences, y_test, verbose=2)
        return acc


class Parallel_CNN_LSTM_Attention_Glove:
    def __init__(self):
        self.embedding_dim = 300
        self.lstm_out = 296
        self.max_words = 350
        self.tokenizer = Tokenizer(num_words=self.max_words, split=' ')
        self.model = Sequential()
        self.label_encoder = LabelEncoder()
        self.glove_path = '/content/drive/MyDrive/glove.6B.300d.txt'

    def load_glove_vectors(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            embeddings_index = {}
            for line in file:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def fit(self, train_texts, train_labels):
        # Tokenize text data
        path = self.glove_path
        glove_vectors = self.load_glove_vectors(path)

        self.tokenizer.fit_on_texts(train_texts)
        X_train = self.tokenizer.texts_to_sequences(train_texts)
        X_train = pad_sequences(X_train, maxlen=self.max_words)
        y_train = self.label_encoder.fit_transform(train_labels)

        num_words = len(self.tokenizer.word_index) + 1
        embedding_matrix = np.zeros((num_words, self.embedding_dim))

        for word, i in self.tokenizer.word_index.items():
            if word in glove_vectors:
                embedding_matrix[i] = glove_vectors[word]

        vocab_size = len(self.tokenizer.word_index) + 1

        embedding_layer = Embedding(vocab_size, self.embedding_dim, weights=[embedding_matrix],
                                    input_length=self.max_words,
                                    trainable=False)

        cnn_model = Sequential()
        cnn_model.add(embedding_layer)
        cnn_model.add(Dropout(0.5))
        cnn_model.add(Conv1D(filters=128,
                             kernel_size=3,
                             padding='same',
                             activation='relu',
                             strides=1))
        cnn_model.add(MaxPooling1D(pool_size=2))
        cnn_model.add(Dropout(0.5))
        cnn_model.add(Conv1D(filters=592,
                             kernel_size=3,
                             padding='same',
                             activation='relu',
                             strides=1))
        cnn_model.add(MaxPooling1D(pool_size=1))
        cnn_model.add(tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=5, strides=2, padding='same'))
        cnn_model.add(Conv1D(filters=592,
                             kernel_size=1,
                             padding='valid',
                             activation='relu',
                             strides=1))

        lstm_model = Sequential()
        lstm_model.add(embedding_layer)
        lstm_model.add(Bidirectional(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        lstm_model.add(Bidirectional(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

        input = tf.keras.layers.Concatenate(axis=0)([cnn_model.output, lstm_model.output])
        attention = SeqSelfAttention(attention_activation='sigmoid')(input)
        average_pooling = GlobalAveragePooling1D()(attention)
        outputs = keras.layers.Dense(len(np.unique(train_labels)), activation='softmax')(average_pooling)
        model = keras.Model(inputs=[cnn_model.output, lstm_model.output], outputs=outputs)

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 32
        epochs = 25
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def predict(self, test_data, test_labels):
        new_sequences = self.tokenizer.texts_to_sequences(test_data)
        new_padded_sequences = pad_sequences(new_sequences)
        y_test = self.label_encoder.fit_transform(test_labels)
        score, acc = self.model.evaluate(new_padded_sequences, y_test, verbose=2)
        return acc


if __name__ == "__main__":
    train_df = pd.read_csv('/content/drive/MyDrive/Train.csv')
    train_df = train_df.sample(frac=1)
    train_texts = train_df['Text'].values
    train_labels = train_df['Label'].values

    test_df = pd.read_csv('/content/drive/MyDrive/Test.csv')
    test_df = test_df.sample(frac=1)
    test_texts = test_df['Text'].values
    test_labels = test_df['Label'].values

    print("Implementing CNN LSTM with Attention")
    model = CNN_LSTM_Attention_Glove()
    model.fit(train_texts, train_labels)
    accuracy = model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    print("Implementing LSTM CNN with Attention")
    model = LSTM_CNN_Attention_Glove()
    model.fit(train_texts, train_labels)
    accuracy = model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    print("Implementing LSTM and CNN in parallel with Attention")
    model = Parallel_CNN_LSTM_Attention_Glove()
    model.fit(train_texts, train_labels)
    accuracy = model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
