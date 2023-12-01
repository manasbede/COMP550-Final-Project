from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Concatenate, Dot, Activation, Embedding, Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import preprocessdata
from sklearn.metrics import accuracy_score


class LSTM_with_Attention():
    def __init__(self):
        self.model = None
        self.tokenizer = Tokenizer(num_words=1000)
        # self.max_sequence_length = 10  # maximum length of input sequences

    def fit(self, train_texts, train_labels):
        # Tokenize text data

        self.tokenizer.fit_on_texts(train_texts)
        sequences = self.tokenizer.texts_to_sequences(train_texts)
        self.max_sequence_length = max(len(seq) for seq in sequences)

        max_words = len(self.tokenizer.word_index) + 1

        # Pad sequences to have the same length
        sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)

        # Convert to numpy arrays
        X = np.array(sequences)
        y = tf.keras.utils.to_categorical(train_labels, num_classes=len(set(train_labels)))

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define model
        latent_dim = 128  # Adjust according to your choice
        input_sequence = Input(shape=(self.max_sequence_length,))
        embedded_sequence = Embedding(max_words, latent_dim)(input_sequence)

        encoder_lstm = Bidirectional(LSTM(units=latent_dim, return_sequences=True))(embedded_sequence)
        decoder_lstm = (LSTM(units=latent_dim, return_sequences=True)(embedded_sequence))

        attention = tf.keras.layers.Attention()([encoder_lstm, encoder_lstm])
        decoder_combined_context = Concatenate(axis=-1)([decoder_lstm, attention])
        output = TimeDistributed(Dense(latent_dim, activation="relu"))(decoder_combined_context)
        output = Flatten()(output)
        output = Dense(len(set(train_labels)), activation='softmax')(output)

        self.model = Model(inputs=input_sequence, outputs=output)

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    def predict(self, test_data):
        new_sequences = self.tokenizer.texts_to_sequences(test_data)
        new_padded_sequences = pad_sequences(new_sequences, maxlen=self.max_sequence_length)
        predictions = self.model.predict(new_padded_sequences)
        return predictions

#   Apply data on LSTM with Attention
train_texts, train_labels = preprocessdata.read_data_from_folders('training')
test_texts, test_labels = preprocessdata.read_data_from_folders('test')

model = LSTM_with_Attention()
model.fit(train_texts,train_labels)
prediction=model.predict(test_texts)
accuracy = accuracy_score(np.asarray(test_labels), prediction)
print(accuracy)