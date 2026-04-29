import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight

def train_tfidf(train_df, test_df):
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X_train = vectorizer.fit_transform(train_df['review'])
    X_test = vectorizer.transform(test_df['review'])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_df['label'])
    preds = model.predict(X_test)

    print("\nTF-IDF Results:")
    print(classification_report(test_df['label'], preds))
    return classification_report(test_df['label'], preds, output_dict=True), preds

def train_transformer(train_df, test_df):
    print("\n--- Training DistilBERT ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(batch):
        return tokenizer(batch["review"], padding="max_length", truncation=True)

    # Keeping mapping flat to ensure features are extracted efficiently
    train_tokenized = train_ds.map(tokenize_fn, batched=True)
    test_tokenized = test_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized
    )

    trainer.train()

    raw_preds = trainer.predict(test_tokenized)
    preds = np.argmax(raw_preds.predictions, axis=1)

    print("\nDistilBERT Results:")
    print(classification_report(test_df['label'], preds))
    return classification_report(test_df['label'], preds, output_dict=True), preds

def train_tfidf_improved(train_df, test_df):
    # Update 1: Added n-grams to catch two-word phrases like "did not"
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['review'])
    X_test = vectorizer.transform(test_df['review'])

    # Update 2: Added class_weight to handle the imbalanced dataset
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, train_df['label'])
    preds = model.predict(X_test)

    print("\nImproved TF-IDF Results:")
    print(classification_report(test_df['label'], preds))
    return classification_report(test_df['label'], preds, output_dict=True), preds

def get_class_weights(df):
    """Calculates balanced class weights as a PyTorch tensor."""
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['label']),
        y=df['label']
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(weights, dtype=torch.float32).to(device)

class WeightedTrainer(Trainer):
    """Subclasses the standard Trainer to apply class weights to the loss function."""
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract labels
        labels = inputs.pop("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Apply weighted Cross Entropy Loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def train_transformer_weighted(train_df, test_df):
    print("\n--- Training Weighted DistilBERT ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(batch):
        return tokenizer(batch["review"], padding="max_length", truncation=True)

    train_tokenized = train_ds.map(tokenize_fn, batched=True)
    test_tokenized = test_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        num_train_epochs=3,
        weight_decay=0.01,
        learning_rate=5e-5,   # Back to a standard, slightly more aggressive rate
        logging_steps=50,
        report_to="none"
    )

    # Calculate weights dynamically based on the training set
    weights_tensor = get_class_weights(train_df)

    # Use our custom Trainer
    trainer = WeightedTrainer(
        class_weights=weights_tensor,
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized
    )

    trainer.train()

    raw_preds = trainer.predict(test_tokenized)
    preds = np.argmax(raw_preds.predictions, axis=1)

    print("\nWeighted DistilBERT Results:")
    print(classification_report(test_df['label'], preds))
    return classification_report(test_df['label'], preds, output_dict=True), preds, trainer.state.log_history