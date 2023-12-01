import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# Define the dataset
reviews = [
    "The product is great and works perfectly!",
    "I am really disappointed with the quality of the item.",
    "This is the best purchase I've ever made!",
    "The customer service was terrible, and I will not recommend this product.",
    "I highly recommend this product to everyone!"
]
sentiments = np.asarray([1, 0, 1, 0, 1])  # 1: Positive, 0: Negative


# Tokenize and preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
vocab_size = len(tokenizer.word_index) + 1


max_sequence_length = 10  # maximum length of input sequences
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)


# Define the LSTM model architecture
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_sequence_length))
model.add(LSTM(units=128))
model.add(Dense(units=1, activation='sigmoid'))


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, sentiments, epochs=10, batch_size=32)


# Make predictions
new_reviews = [
    "The product exceeded my expectations!",
    "I regret buying this item."
]
new_sequences = tokenizer.texts_to_sequences(new_reviews)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
predictions = model.predict(new_padded_sequences)


# Convert predictions to sentiment labels
sentiment_labels = ['Negative', 'Positive']
predicted_labels = [sentiment_labels[int(np.round(pred))] for pred in predictions]

# Print the new reviews and predicted sentiment labels
for i in range(len(new_reviews)):
    print('Review:', new_reviews[i])
    print('Predicted Sentiment:', predicted_labels[i])
    print('---')
