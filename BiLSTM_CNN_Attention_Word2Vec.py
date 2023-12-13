import nltk

nltk.download('punkt')

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling1D
from keras_self_attention import SeqSelfAttention  # Import the attention layer
import pandas as pd
from keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


class BiLSTM_CNN_with_Attention_W2Vec:
    def __init__(self):
        self.embedding_dim = 300
        self.lstm_out = 296
        self.max_words = 10000
        self.tokenizer = Tokenizer(num_words=self.max_words, split=' ')
        self.model = Sequential()
        self.label_encoder = LabelEncoder()

    def fit(self, train_texts, train_labels):
        word_vectors = Word2Vec.load('/content/word2vec_weights.bin')

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_texts)

        vocab_size = len(tokenizer.word_index) + 1

        # Convert text to sequences
        sequences = tokenizer.texts_to_sequences(train_texts)

        # Padding sequences to a fixed length
        max_length = 2000  # Adjust as needed
        padded_sequences = pad_sequences(sequences, maxlen=max_length)

        # Convert labels to one-hot encoding
        num_classes = len(np.unique(train_labels))
        y_train_one_hot = np.eye(num_classes)[train_labels]

        # Embedding layer using pretrained Word2Vec weights
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, i in tokenizer.word_index.items():
            if word in word_vectors.wv:
                embedding_matrix[i] = word_vectors.wv[word]

        embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length,
                                    trainable=False)
        self.model.add(embedding_layer)
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(filters=128,
                              kernel_size=8,
                              padding='valid',
                              activation='relu',
                              strides=1))

        self.model.add(MaxPooling1D(pool_size=4))

        # Add LSTM layer with return_sequences=True to get the sequence output
        self.model.add(Bidirectional(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

        # Add Attention layer
        self.model.add(SeqSelfAttention(attention_activation='sigmoid'))

        # Use GlobalAveragePooling1D to get a fixed-size output regardless of the sequence length
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(len(np.unique(train_labels)), activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 32
        epochs = 10
        self.model.fit(padded_sequences, y_train_one_hot, epochs=epochs, batch_size=batch_size, validation_split=0.1)

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

    print("Implementing BiLSTM with CNN and Attention and Word2Vec embedding Model")
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in train_texts]

    # Define Word2Vec model parameters
    vector_size = 2000  # Dimensionality of word vectors
    window_size = 5  # Context window size
    min_count = 1  # Minimum word frequency threshold
    workers = 4  # Number of threads to use for training

    # Train the Word2Vec model
    word2vec_model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window_size,
        min_count=min_count,
        workers=workers
    )
    word2vec_model.save("word2vec_model.bin")

    # Load pre-trained Word2Vec model
    word2vec_model = Word2Vec.load('/content/drive/MyDrive/word2vec_model.bin')
    embedding_dim = word2vec_model.vector_size

    model = BiLSTM_CNN_with_Attention_W2Vec()
    model.fit(train_texts, train_labels)
    accuracy = model.predict(test_texts, test_labels)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
