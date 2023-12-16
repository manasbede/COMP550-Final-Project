import pandas as pd
from sklearn.preprocessing import LabelEncoder
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_self_attention import SeqSelfAttention
from keras import utils
import tensorflow as tf

if __name__ == "__main__":
    train_df = pd.read_csv('D:\\Fall2023\\COMP550 NLP\\Project\\NLP550Project\\Train.csv')
    train_df = train_df.sample(frac=1)
    train_texts = train_df['Text'].values
    train_labels = train_df['Label'].values

    test_df = pd.read_csv('D:\\Fall2023\\COMP550 NLP\\Project\\NLP550Project\\Test.csv')
    test_df = test_df.sample(frac=1)
    test_texts = test_df['Text'].values
    test_labels = test_df['Label'].values

    word_counts = [len(sentence.split()) for sentence in train_texts]
    top_samples_count = int(len(word_counts) * 0.1)
    top_samples_indices = sorted(range(len(word_counts)), key=lambda i: word_counts[i], reverse=True)[
                          :top_samples_count]
    train_texts_top10 = [train_texts[i] for i in top_samples_indices]
    train_labels_top10 = [train_labels[i] for i in top_samples_indices]

    word_counts = [len(sentence.split()) for sentence in test_texts]
    top_samples_count = int(len(word_counts) * 0.1)
    top_samples_indices = sorted(range(len(word_counts)), key=lambda i: word_counts[i], reverse=True)[
                          :top_samples_count]
    test_texts_top10 = [test_texts[i] for i in top_samples_indices]
    test_labels_top10 = [test_labels[i] for i in top_samples_indices]

    # print("Evaluation on simple LSTM")
    # tokenizer = Tokenizer(num_words=5000, split=' ')
    # label_encoder = LabelEncoder()
    # tokenizer.fit_on_texts(train_texts)
    # train_texts_tokenized = tokenizer.texts_to_sequences(train_texts)
    # max_sequence_length = max(len(seq) for seq in train_texts_tokenized)
    #
    # loaded_model = keras.models.load_model(
    #     "D:\\Fall2023\\COMP550 NLP\\Project\\NLP550Project\\trained-models\\lstm.keras")
    #
    # new_sequences = tokenizer.texts_to_sequences(train_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    # y_test = label_encoder.fit_transform(train_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Train set accuracy : ", acc)
    #
    # new_sequences = tokenizer.texts_to_sequences(test_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    # y_test = label_encoder.fit_transform(test_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Test set accuracy : ", acc)
    #
    # print("Evaluation on LSTM with Glove word embedding")
    # tokenizer = Tokenizer(num_words=10000, split=' ')
    # label_encoder = LabelEncoder()
    # tokenizer.fit_on_texts(train_texts)
    # train_texts_tokenized = tokenizer.texts_to_sequences(train_texts)
    # max_sequence_length = max(len(seq) for seq in train_texts_tokenized)
    #
    # loaded_model = keras.models.load_model(
    #     "D:\\Fall2023\\COMP550 NLP\\Project\\NLP550Project\\trained-models\\lstm_we.keras")
    #
    # new_sequences = tokenizer.texts_to_sequences(train_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    # y_test = label_encoder.fit_transform(train_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Train set accuracy : ", acc)
    #
    # new_sequences = tokenizer.texts_to_sequences(test_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    # y_test = label_encoder.fit_transform(test_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Test set accuracy : ", acc)
    #
    # print("Evaluation on LSTM with Attention")
    # tokenizer = Tokenizer(num_words=5000, split=' ')
    # label_encoder = LabelEncoder()
    # tokenizer.fit_on_texts(train_texts)
    # train_texts_tokenized = tokenizer.texts_to_sequences(train_texts)
    # max_sequence_length = max(len(seq) for seq in train_texts_tokenized)
    #
    # loaded_model = keras.models.load_model(
    #     "D:\\Fall2023\\COMP550 NLP\\Project\\NLP550Project\\trained-models\\lstm_att.keras",
    #     custom_objects=SeqSelfAttention.get_custom_objects())
    #
    # new_sequences = tokenizer.texts_to_sequences(train_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    # y_test = label_encoder.fit_transform(train_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Train set accuracy : ", acc)
    #
    # new_sequences = tokenizer.texts_to_sequences(test_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    # y_test = label_encoder.fit_transform(test_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Test set accuracy : ", acc)

    # print("Evaluation on LSTM with glove word embedding and Attention")
    # tokenizer = Tokenizer(num_words=10000, split=' ')
    # label_encoder = LabelEncoder()
    # tokenizer.fit_on_texts(train_texts)
    # train_texts_tokenized = tokenizer.texts_to_sequences(train_texts)
    # max_sequence_length = max(len(seq) for seq in train_texts_tokenized)
    #
    # loaded_model = keras.models.load_model(
    #     "D:\\Fall2023\\COMP550 NLP\\Project\\NLP550Project\\trained-models\\lstm_att_we.keras",
    #     custom_objects=SeqSelfAttention.get_custom_objects())
    #
    # new_sequences = tokenizer.texts_to_sequences(train_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    # y_test = label_encoder.fit_transform(train_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Train set accuracy : ", acc)
    #
    # new_sequences = tokenizer.texts_to_sequences(test_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    # y_test = label_encoder.fit_transform(test_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Test set accuracy : ", acc)

    # print("Evaluation on CNN LSTM")
    # tokenizer = Tokenizer(num_words=1000, split=' ')
    # label_encoder = LabelEncoder()
    # tokenizer.fit_on_texts(train_texts)
    # train_texts_tokenized = tokenizer.texts_to_sequences(train_texts)
    # max_sequence_length = max(len(seq) for seq in train_texts_tokenized)
    #
    # loaded_model = keras.models.load_model(
    #     "D:\\Fall2023\\COMP550 NLP\\Project\\NLP550Project\\trained-models\\CNN_LSTM.keras")
    #
    # new_sequences = tokenizer.texts_to_sequences(train_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=1000)
    # y_test = utils.to_categorical(train_labels_top10, num_classes=23)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Train set accuracy : ", acc)
    #
    # new_sequences = tokenizer.texts_to_sequences(test_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=1000)
    # y_test = utils.to_categorical(test_labels_top10, num_classes=23)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Test set accuracy : ", acc)

    # print("Evaluation on BiLSTM with Attention")
    # tokenizer = Tokenizer(num_words=5000, split=' ')
    # label_encoder = LabelEncoder()
    # tokenizer.fit_on_texts(train_texts)
    # train_texts_tokenized = tokenizer.texts_to_sequences(train_texts)
    # max_sequence_length = max(len(seq) for seq in train_texts_tokenized)
    #
    # loaded_model = keras.models.load_model(
    #     "D:\\Fall2023\\COMP550 NLP\\Project\\NLP550Project\\trained-models\\bilstm_att.keras",
    #     custom_objects=SeqSelfAttention.get_custom_objects())
    #
    # new_sequences = tokenizer.texts_to_sequences(train_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    # y_test = label_encoder.fit_transform(train_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Train set accuracy : ", acc)
    #
    # new_sequences = tokenizer.texts_to_sequences(test_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    # y_test = label_encoder.fit_transform(test_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Test set accuracy : ", acc)

    # print("Evaluation on BiLSTM with word embedding and Attention")
    # tokenizer = Tokenizer(num_words=10000, split=' ')
    # label_encoder = LabelEncoder()
    # tokenizer.fit_on_texts(train_texts)
    # train_texts_tokenized = tokenizer.texts_to_sequences(train_texts)
    # max_sequence_length = max(len(seq) for seq in train_texts_tokenized)
    #
    # loaded_model = keras.models.load_model(
    #     "D:\\Fall2023\\COMP550 NLP\\Project\\NLP550Project\\trained-models\\bilstm_att_we.keras",
    #     custom_objects=SeqSelfAttention.get_custom_objects())
    #
    # new_sequences = tokenizer.texts_to_sequences(train_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=356)
    # y_test = label_encoder.fit_transform(train_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Train set accuracy : ", acc)
    #
    # new_sequences = tokenizer.texts_to_sequences(test_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=356)
    # y_test = label_encoder.fit_transform(test_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Test set accuracy : ", acc)

    # print("Evaluation on BiLSTM CNN model with glove word embedding and Attention")
    # tokenizer = Tokenizer(num_words=5000, split=' ')
    # label_encoder = LabelEncoder()
    # tokenizer.fit_on_texts(train_texts)
    # train_texts_tokenized = tokenizer.texts_to_sequences(train_texts)
    # max_sequence_length = max(len(seq) for seq in train_texts_tokenized)
    #
    # loaded_model = keras.models.load_model(
    #     "D:\\Fall2023\\COMP550 NLP\\Project\\NLP550Project\\trained-models\\bilstm_cnn_att_we.keras",
    #     custom_objects=SeqSelfAttention.get_custom_objects())
    #
    # new_sequences = tokenizer.texts_to_sequences(train_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=350)
    # y_test = label_encoder.fit_transform(train_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Train set accuracy : ", acc)
    #
    # new_sequences = tokenizer.texts_to_sequences(test_texts_top10)
    # new_padded_sequences = pad_sequences(new_sequences, maxlen=350)
    # y_test = label_encoder.fit_transform(test_labels_top10)
    # score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    # print("Test set accuracy : ", acc)

    print("Evaluation on BiLSTM CNN model with Word2Vec word embedding and Attention")
    tokenizer = Tokenizer(num_words=5000, split=' ')
    label_encoder = LabelEncoder()
    tokenizer.fit_on_texts(train_texts)
    train_texts_tokenized = tokenizer.texts_to_sequences(train_texts)
    max_sequence_length = max(len(seq) for seq in train_texts_tokenized)

    loaded_model = keras.models.load_model(
        "D:\\Fall2023\\COMP550 NLP\\Project\\NLP550Project\\trained-models\\bilstm_cnn_att_w2v_we.keras",
        custom_objects=SeqSelfAttention.get_custom_objects())

    new_sequences = tokenizer.texts_to_sequences(train_texts_top10)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=350)
    y_test = label_encoder.fit_transform(train_labels_top10)
    score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    print("Train set accuracy : ", acc)

    new_sequences = tokenizer.texts_to_sequences(test_texts_top10)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=350)
    y_test = label_encoder.fit_transform(test_labels_top10)
    score, acc = loaded_model.evaluate(new_padded_sequences, y_test, verbose=2)
    print("Test set accuracy : ", acc)
