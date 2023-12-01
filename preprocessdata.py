import numpy as np
import os
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from nltk.stem import WordNetLemmatizer

def read_data_from_folders(directory):
    root_folder=os.getcwd()
    root_folder=root_folder+f"\\ohsumed-first-20000-docs\\ohsumed-first-20000-docs\\{directory}"
    data = []
    labels = []

    label_folders = sorted(os.listdir(root_folder))  # Get sorted list of label folders

    for label, folder_name in enumerate(label_folders):
        label_folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(label_folder_path):
            file_names = os.listdir(label_folder_path)
            for file_name in file_names:
                file_path = os.path.join(label_folder_path, file_name)
                if os.path.isfile(file_path) :
                    with open(file_path, 'r') as file:
                        text = file.read()
                        data.append(text)
                        labels.append(label)

    return list(data), list(labels)

def remove_stop_words(data):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(data)
    filtered_paragraph = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_paragraph)

def remove_punctuation_blank_space(data):
    translator = str.maketrans("", "", string.punctuation + "[](){}")
    data=data.lower()
    cleaned_text = data.translate(translator)
    cleaned_text = re.sub(' +', ' ', cleaned_text.strip())
    return cleaned_text

def lemmatization(data):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(data)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in word_tokens]
    lemma_text=' '.join(lemmatized_words)
    cleaned_text = ' '.join(lemma_text.split())
    return cleaned_text

def pre_process(data_list):
    data_list_after_punct_remove = [remove_punctuation_blank_space(data) for data in data_list]
    data_after_stop_word_remove = [remove_stop_words(data) for data in data_list_after_punct_remove]
    print(data_after_stop_word_remove[0])
    data_after_lemmatization = [lemmatization(data) for data in data_after_stop_word_remove]
    print()
    print(data_after_lemmatization[0])
    

if __name__ == "__main__":
    train_texts, train_labels = read_data_from_folders('training')
    test_texts, test_labels = read_data_from_folders('test')
    pre_process(train_texts)