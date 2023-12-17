import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv('Train.csv')
train_df = train_df.sample(frac=1)
train_texts = train_df['Text'].values
train_labels = train_df['Label'].values

test_df = pd.read_csv('Test.csv')
test_df = test_df.sample(frac=1)
test_texts = test_df['Text'].values
test_labels = test_df['Label'].values

# Dataset distribution of each disease samples
unique_elements = list(set(train_labels))
frequency = [list(train_labels).count(elem) for elem in unique_elements]
categories = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13',
              'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23']
plt.bar(categories, frequency)
plt.xlabel('Disease category')
plt.ylabel('Frequency')
plt.show()


# Sentence length distribution
train_df['Length'] = train_df['Text'].apply(lambda x: len(x.split()))

# Count the number of sentences for each length
sentence_lengths = train_df['Length'].value_counts().sort_index()

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(sentence_lengths.index, sentence_lengths.values, color='skyblue')
plt.xlabel('Length of Sentences in terms of word')
plt.ylabel('Number of Sentences')
plt.grid(axis='y')
plt.show()



