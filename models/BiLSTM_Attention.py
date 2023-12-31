import numpy as np
import pandas as pd
from keras.src.layers import Bidirectional
from keras_self_attention import SeqSelfAttention  # Import the attention layer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class BiLSTM_with_Attention():
    def __init__(self):
        self.embedding_dim = 128
        self.lstm_out = 196
        self.max_words = 5000
        self.tokenizer = Tokenizer(num_words=self.max_words, split=' ')
        self.model = Sequential()
        self.label_encoder = LabelEncoder()

    def fit(self, train_texts, train_labels):
        # Tokenize text data

        self.tokenizer.fit_on_texts(train_texts)
        X_train = self.tokenizer.texts_to_sequences(train_texts)
        self.max_sequence_length = max(len(seq) for seq in X_train)

        #max_words = len(self.tokenizer.word_index) + 1

        # Pad sequences to have the same length
        X_train = pad_sequences(X_train)

        y_train = self.label_encoder.fit_transform(train_labels)

        self.model.add(Embedding(self.max_words, self.embedding_dim, input_length=X_train.shape[1]))
        self.model.add(SpatialDropout1D(0.2))

        # Add LSTM layer with return_sequences=True to get the sequence output
        self.model.add(Bidirectional(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

        # Add Attention layer
        self.model.add(SeqSelfAttention(attention_activation='sigmoid'))

        # Use GlobalAveragePooling1D to get a fixed-size output regardless of the sequence length
        self.model.add(GlobalAveragePooling1D())

        self.model.add(Dense(len(np.unique(y_train)), activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 32
        epochs = 4

        # checkpoint_filepath = '/content/drive/MyDrive/COMP550/checkpoint.model'
        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     monitor='val_accuracy',
        #     mode='max',
        #     save_best_only=True)

        # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[model_checkpoint_callback])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def predict(self, test_data, test_labels):
        new_sequences = self.tokenizer.texts_to_sequences(test_data)
        new_padded_sequences = pad_sequences(new_sequences)
        y_test = self.label_encoder.fit_transform(test_labels)
        score, acc = self.model.evaluate(new_padded_sequences, y_test, verbose=2)
        return acc

    def save(self):
        self.model.save("lstm_att.keras")


class BiLSTM_with_Attention_WE():
    def __init__(self):
        self.embedding_dim = 300
        self.lstm_out = 296
        self.max_words = 10000
        self.tokenizer = Tokenizer(num_words=self.max_words, split=' ')
        self.model = Sequential()
        self.label_encoder = LabelEncoder()
        self.glove_path = 'glove.6B.300d.txt'

    def load_glove_vectors(self,file_path):
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
        path=self.glove_path
        glove_vectors = self.load_glove_vectors(path)

        self.tokenizer.fit_on_texts(train_texts)
        X_train = self.tokenizer.texts_to_sequences(train_texts)
        X_train = pad_sequences(X_train)
        y_train = self.label_encoder.fit_transform(train_labels)

        num_words = len(self.tokenizer.word_index) + 1
        embedding_matrix = np.zeros((num_words, self.embedding_dim))

        for word, i in self.tokenizer.word_index.items():
            if word in glove_vectors:
                embedding_matrix[i] = glove_vectors[word]

        self.model.add(Embedding(num_words, self.embedding_dim, weights=[embedding_matrix], input_length=X_train.shape[1], trainable=False))
        self.model.add(SpatialDropout1D(0.2))

        # Add LSTM layer with return_sequences=True to get the sequence output
        self.model.add(Bidirectional(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

        # Add Attention layer
        self.model.add(SeqSelfAttention(attention_activation='sigmoid'))

        # Use GlobalAveragePooling1D to get a fixed-size output regardless of the sequence length
        self.model.add(GlobalAveragePooling1D())

        self.model.add(Dense(len(np.unique(y_train)), activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 32
        epochs = 4

        # checkpoint_filepath = '/content/drive/MyDrive/COMP550/checkpoint.model'
        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     monitor='val_accuracy',
        #     mode='max',
        #     save_best_only=True)

        # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[model_checkpoint_callback])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def predict(self, test_data, test_labels):
        new_sequences = self.tokenizer.texts_to_sequences(test_data)
        new_padded_sequences = pad_sequences(new_sequences)
        y_test = self.label_encoder.fit_transform(test_labels)
        score, acc = self.model.evaluate(new_padded_sequences, y_test, verbose=2)
        return acc

    def save(self):
        self.model.save("lstm_att_we.keras")

if __name__ == "__main__":
    train_df=pd.read_csv('../Train.csv')
    train_df = train_df.sample(frac = 1)
    train_texts=train_df['Text'].values
    train_labels=train_df['Label'].values

    test_df=pd.read_csv('../Test.csv')
    test_df = test_df.sample(frac = 1)
    test_texts=test_df['Text'].values
    test_labels=test_df['Label'].values

    print("Implementing BiLSTM with Attention Model")
    model = BiLSTM_with_Attention()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    print("Implementing BiLSTM with Attention and Word Embedding Model")
    model = BiLSTM_with_Attention_WE()
    model.fit(train_texts,train_labels)
    accuracy=model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')