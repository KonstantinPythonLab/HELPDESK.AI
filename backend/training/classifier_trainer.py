"""
Classifier Trainer — Multi-Output Classification
Fine-tunes distilbert-base-uncased for:
  category, sub_category, priority, auto_resolve, assigned_team
"""

import os
import sys
import glob
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")
SAVE_DIR = os.path.join(PROJECT_ROOT, "backend", "models", "classifier")

LABEL_COLUMNS = ["category", "sub_category", "priority", "auto_resolve", "assigned_team"]
TEXT_COLUMN = "text"

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Auto-detect dataset
# ---------------------------------------------------------------------------
def _find_dataset() -> str:
    """Find Excel or CSV inside the Model folder."""
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        matches = glob.glob(os.path.join(MODEL_DIR, ext))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No Excel/CSV dataset found in {MODEL_DIR}")


def _load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# Custom PyTorch Dataset
# ---------------------------------------------------------------------------
class TicketDataset(TorchDataset):
    def __init__(self, encodings, labels_dict):
        self.encodings = encodings
        self.labels_dict = labels_dict  # {col_name: np.array}

    def __len__(self):
        return len(self.labels_dict[LABEL_COLUMNS[0]])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        for col in LABEL_COLUMNS:
            item[f"label_{col}"] = torch.tensor(self.labels_dict[col][idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Multi-Output Model
# ---------------------------------------------------------------------------
class MultiOutputClassifier(nn.Module):
    def __init__(self, num_labels_per_output: dict):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden = self.bert.config.hidden_size  # 768
        self.heads = nn.ModuleDict()
        for name, n_labels in num_labels_per_output.items():
            self.heads[name] = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = {}
        for name, head in self.heads.items():
            logits[name] = head(cls)
        return logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_classifier():
    print("=" * 60)
    print("CLASSIFIER TRAINING")
    print("=" * 60)

    # 1. Load data
    path = _find_dataset()
    print(f"[INFO] Dataset found: {path}")
    df = _load_dataset(path)
    print(f"[INFO] Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Rename column if needed
    col_map = {"user_input_text": "text"}
    df.rename(columns=col_map, inplace=True)

    # Ensure required columns
    required = [TEXT_COLUMN] + LABEL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Drop missing rows
    df.dropna(subset=required, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] After dropping NaN: {len(df)} rows")

    # 2. Encode labels
    label_encoders = {}
    encoded_labels = {}
    for col in LABEL_COLUMNS:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        encoded_labels[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"[INFO] {col}: {len(le.classes_)} classes")

    num_labels_per_output = {col: len(le.classes_) for col, le in label_encoders.items()}

    # 3. Tokenize
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    texts = df[TEXT_COLUMN].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")

    # 4. Train / test split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    def _subset(enc, idx):
        return {k: v[idx] for k, v in enc.items()}

    train_labels = {col: encoded_labels[col][train_idx] for col in LABEL_COLUMNS}
    test_labels = {col: encoded_labels[col][test_idx] for col in LABEL_COLUMNS}

    train_ds = TicketDataset(_subset(encodings, train_idx), train_labels)
    test_ds = TicketDataset(_subset(encodings, test_idx), test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print(f"[INFO] Train: {len(train_ds)}, Test: {len(test_ds)}")

    # 5. Model
    model = MultiOutputClassifier(num_labels_per_output).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Training for {EPOCHS} epochs …")

    # 6. Train
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = sum(
                loss_fn(logits[col], batch[f"label_{col}"].to(DEVICE))
                for col in LABEL_COLUMNS
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}")

    # 7. Evaluate
    print("\n[EVAL] Computing metrics on test set …")
    model.eval()
    all_preds = {col: [] for col in LABEL_COLUMNS}
    all_trues = {col: [] for col in LABEL_COLUMNS}

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids, attention_mask)
            for col in LABEL_COLUMNS:
                preds = torch.argmax(logits[col], dim=1).cpu().numpy()
                trues = batch[f"label_{col}"].numpy()
                all_preds[col].extend(preds)
                all_trues[col].extend(trues)

    for col in LABEL_COLUMNS:
        acc = accuracy_score(all_trues[col], all_preds[col])
        f1 = f1_score(all_trues[col], all_preds[col], average="weighted")
        print(f"  {col:20s}  accuracy={acc:.4f}  weighted_f1={f1:.4f}")

    # 8. Save
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
    tokenizer.save_pretrained(SAVE_DIR)

    # Save label encoders
    with open(os.path.join(SAVE_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)

    # Save model config (num labels per output)
    with open(os.path.join(SAVE_DIR, "model_config.json"), "w") as f:
        json.dump(num_labels_per_output, f)

    print(f"\n[INFO] Model saved to {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train_classifier()
