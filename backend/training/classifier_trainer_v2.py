"""
Classifier Trainer V2 — Shadow Model Pipeline
Optimized for Pragati's 10k Dataset with longer context.
"""

import os
import pickle
import json
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
# Configuration (Optimized for Phase 2)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")
SAVE_DIR = os.path.join(PROJECT_ROOT, "backend", "models", "classifier-v2") # Shadow Folder
DATASET_PATH = os.path.join(MODEL_DIR, "Final_Balanced_10000_IT_Support_Tickets.csv")

LABEL_COLUMNS = ["category", "sub_category", "Priority", "auto_resolve", "assigned_team"]
TEXT_COLUMN = "user_input_text"

MAX_LEN = 256  # Increased from 128 for longer text handling
BATCH_SIZE = 16 # Reduced batch size for higher MAX_LEN to fit in T4/Pro VRAM
EPOCHS = 4      # Slightly more epochs for 10k rows
LEARNING_RATE = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Dataset & Model
# ---------------------------------------------------------------------------
class TicketDataset(TorchDataset):
    def __init__(self, encodings, labels_dict):
        self.encodings = encodings
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.labels_dict[LABEL_COLUMNS[0]])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        for col in LABEL_COLUMNS:
            item[f"label_{col}"] = torch.tensor(self.labels_dict[col][idx], dtype=torch.long)
        return item

class MultiOutputClassifierV2(nn.Module):
    def __init__(self, num_labels_per_output: dict):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden = self.bert.config.hidden_size 
        
        # Optimization: Added dropout for better generalization on synthetic data
        self.dropout = nn.Dropout(0.2)
        
        self.heads = nn.ModuleDict()
        for name, n_labels in num_labels_per_output.items():
            self.heads[name] = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # We use the [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0] 
        cls_output = self.dropout(cls_output)
        
        logits = {name: head(cls_output) for name, head in self.heads.items()}
        return logits

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------
def train_v2():
    print("🚀 Starting SHADOW TRAINING (V2)")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"[INFO] Loaded {len(df)} rows.")

    # 1. Preprocessing
    df.dropna(subset=[TEXT_COLUMN] + LABEL_COLUMNS, inplace=True)
    df.reset_index(drop=True, inplace=True)

    label_encoders = {}
    encoded_labels = {}
    for col in LABEL_COLUMNS:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        encoded_labels[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f" - {col}: {len(le.classes_)} classes")

    num_labels_per_output = {col: len(le.classes_) for col, le in label_encoders.items()}

    # 2. Tokenization
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    texts = df[TEXT_COLUMN].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")

    # 3. Split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42) # Slightly larger train set

    def _subset(enc, idx):
        return {k: v[idx] for k, v in enc.items()}

    train_labels = {col: encoded_labels[col][train_idx] for col in LABEL_COLUMNS}
    test_labels = {col: encoded_labels[col][test_idx] for col in LABEL_COLUMNS}

    train_ds = TicketDataset(_subset(encodings, train_idx), train_labels)
    test_ds = TicketDataset(_subset(encodings, test_idx), test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 4. Model Setup
    model = MultiOutputClassifierV2(num_labels_per_output).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    # 5. Training Loop
    print(f"\n[TRAIN] Training on {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = sum(loss_fn(logits[col], batch[f"label_{col}"].to(DEVICE)) for col in LABEL_COLUMNS)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (i+1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.4f}")

    # 6. Evaluation
    print("\n[EVAL] Comparing V2 accuracy...")
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
                all_preds[col].extend(preds)
                all_trues[col].extend(batch[f"label_{col}"].numpy())

    results = {}
    for col in LABEL_COLUMNS:
        acc = accuracy_score(all_trues[col], all_preds[col])
        results[col] = acc
        print(f"  Accuracy [{col}]: {acc:.4f}")

    # 7. Save to Shadow Folder
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
    tokenizer.save_pretrained(SAVE_DIR)
    
    with open(os.path.join(SAVE_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)
    
    with open(os.path.join(SAVE_DIR, "model_config.json"), "w") as f:
        json.dump(num_labels_per_output, f)
        
    print(f"\n[SUCCESS] V2 Model saved to {SAVE_DIR}")

if __name__ == "__main__":
    train_v2()
