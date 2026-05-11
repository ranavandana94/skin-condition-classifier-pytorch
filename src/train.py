import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataset import SkinDataset
from torch.utils.data import Subset

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/metadata.csv"
IMG_DIR = "data/images"
MODEL_PATH = "models/model.pth"
BATCH_SIZE = 4
EPOCHS = 3
LR = 0.001

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

# -----------------------------
# Load + preprocess data
# -----------------------------
df = pd.read_csv(DATA_PATH)


df["image"] = df["image_id"] + ".jpg"

le = LabelEncoder()
df["label"] = le.fit_transform(df["dx"])

df = df[["image", "label"]]

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)


subset_size = 500
train_df = train_df.iloc[:subset_size]
val_df = val_df.iloc[:500]

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Dataset + DataLoader
# -----------------------------
train_dataset = SkinDataset(train_df, IMG_DIR, transform)
val_dataset = SkinDataset(val_df, IMG_DIR, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# Model (Transfer Learning)
# -----------------------------
model = models.resnet18(weights=None)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 7)


model = model.to(device)

# -----------------------------
# Loss + Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Add batch loss
        train_loss += loss.item()


        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f}")
        

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("-" * 30)

# -----------------------------
# Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)

print("Model saved to", MODEL_PATH)

