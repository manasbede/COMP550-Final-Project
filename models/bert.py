import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from transformers import BertTokenizer, BertForSequenceClassification


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERT:
    def __init__(self, max_len, batch, epochs, learning_rate):
        self.max_len = max_len
        self.batch = batch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=23)

    def fit(self, train_loader):
        self.bert_model.to(self.device)

        optimizer = AdamW(self.bert_model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            self.bert_model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.bert_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}")

        torch.save(self.bert_model.state_dict(), 'weights')

    def predict(self, test_loader):
        predictions = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            predictions.append(preds)
        return predictions

    def preprocess(self, texts, labels):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dataset = CustomDataset(texts, labels, tokenizer, self.max_len)
        dataset_loader = DataLoader(dataset, batch_size=self.batch, shuffle=True)
        return dataset_loader

    def chunk_text(self, text, max_len):
        chunks = []
        words = text.split()
        for i in range(0, len(words), max_len):
            chunk = " ".join(words[i:i + max_len])
            chunks.append(chunk)
        return chunks

    def preprocess_with_chunking(self, texts, labels):
        chunked_texts = [self.chunk_text(text, self.max_len) for text in texts]
        chunked_labels = [[label] * len(chunks) for label, chunks in zip(labels, chunked_texts)]
        chunked_texts = [item for sublist in chunked_texts for item in sublist]
        chunked_labels = [item for sublist in chunked_labels for item in sublist]

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dataset = CustomDataset(chunked_texts, chunked_labels, tokenizer, self.max_len)
        dataset_loader = DataLoader(dataset, batch_size=self.batch, shuffle=True)

        return dataset_loader


if __name__ == "__main__":
    train_df = pd.read_csv('../Train.csv')
    train_texts = train_df['Text'].values
    train_labels = train_df['Label'].values

    test_df = pd.read_csv('../Test.csv')
    test_texts = test_df['Text'].values
    test_labels = test_df['Label'].values

    num_classes = len(set(train_texts))

    print("Implementing BERT model by truncating maximum length of sentence to 512")
    model = BERT(512, 8, 10, 2e-5)
    train_dataset_loader = model.preprocess(train_texts, train_labels)
    test_dataset_loader = model.preprocess(test_texts, test_labels)
    model.fit(train_dataset_loader)
    predicted_label = model.predict(test_dataset_loader)
    accuracy = accuracy_score(np.asarray(test_labels), np.asarray(predicted_label))
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    print("Implementing BERT model by splitting one sentence into multiple sentences to maintain maximum length of 512")
    model = BERT(512, 8, 10, 2e-5)
    train_dataset_loader = model.preprocess_with_chunking(train_texts, train_labels)
    test_dataset_loader = model.preprocess_with_chunking(test_texts, test_labels)
    model.fit(train_dataset_loader)
    predicted_label = model.predict(test_dataset_loader)
    accuracy = accuracy_score(np.asarray(test_labels), np.asarray(predicted_label))
    print(f'Test accuracy: {accuracy * 100:.2f}%')
