import json
import re
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from os.path import join, dirname, abspath
import torch
from torch.utils.data import Dataset


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

class BertTextClassifier:
    def __init__(self, dataset_path=None, model_path=None):
        self.dataset_path = dataset_path or join(dirname(abspath(__file__)), 'data', 'dataset.json')
        self.model_path = model_path or 'bert_model'
        self.model = None
        self.tokenizer = None
        self.stop_words = set(['акт', 'лист', 'приказ', 'распоряжение', '№'])

    def preprocess_text(self, text):
        text = re.sub(r"[^а-яА-Яa-zA-Z0-9.,!?\s]", "", text)  # Удаление лишних символов
        text = text.lower()  # Приведение к нижнему регистру
        words = text.split()
        words = [word for word in words if word not in self.stop_words]  # Удаление стоп-слов
        return ' '.join(words)

    def load_data(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [item["text"] for item in data]
        labels = [1 if item["label"] == "historical_background" else 0 for item in data]
        return texts, labels

    def encode_texts(self, texts, labels, tokenizer, max_length=128):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)
        }

    def train(self):
        texts, labels = self.load_data()
        texts = [self.preprocess_text(t) for t in texts]
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_encodings = self.encode_texts(train_texts, train_labels, self.tokenizer)
        test_encodings = self.encode_texts(test_texts, test_labels, self.tokenizer)

        train_dataset = TextDataset(train_encodings)
        test_dataset = TextDataset(test_encodings)

        self.model = BertModel.from_pretrained('bert-base-uncased', num_labels=2)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        trainer.train()

    def save_model(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

    def load_model(self):
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)

    def predict(self, text):
        if not self.model or not self.tokenizer:
            self.load_model()
        preprocessed_text = self.preprocess_text(text)
        inputs = self.tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs.argmax().item()

    def predict_proba(self, text):
        if not self.model or not self.tokenizer:
            self.load_model()
        preprocessed_text = self.preprocess_text(text)
        inputs = self.tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs.detach().numpy()[0]

    def evaluate(self):
        texts, labels = self.load_data()
        texts = [self.preprocess_text(t) for t in texts]
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
        test_encodings = self.encode_texts(test_texts, test_labels, self.tokenizer)

        test_dataset = TextDataset(test_encodings)

        trainer = Trainer(model=self.model)
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        probs = predictions.predictions[:, 1]

        print("Accuracy:", accuracy_score(test_labels, preds))

    def plot_confusion_matrix(self):
        texts, labels = self.load_data()
        texts = [self.preprocess_text(t) for t in texts]
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
        test_encodings = self.encode_texts(test_texts, test_labels, self.tokenizer)

        test_dataset = TextDataset(test_encodings)

        trainer = Trainer(model=self.model)
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)

        cm = confusion_matrix(test_labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Twaddle', 'Historical Background'], yticklabels=['Twaddle', 'Historical Background'])
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Фактические значения')
        plt.title('Матрица ошибок (BERT)')
        plt.show()

    def plot_roc_curve(self):
        texts, labels = self.load_data()
        texts = [self.preprocess_text(t) for t in texts]
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
        test_encodings = self.encode_texts(test_texts, test_labels, self.tokenizer)

        test_dataset = TextDataset(test_encodings)

        trainer = Trainer(model=self.model)
        predictions = trainer.predict(test_dataset)
        probs = predictions.predictions[:, 1]

        fpr, tpr, thresholds = roc_curve(test_labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривая (BERT)')
        plt.legend(loc="lower right")
        plt.show()