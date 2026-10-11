import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from MidtermDatasetBoth import (
    train_loader,
    val_loader,
    test_loader,
    vocab,
    LABEL_NAMES,
    PAD_IDX,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

NUM_CLASSES = 4

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

NUM_LAYERS = 1
DROPOUT = 0.5

LEARNING_RATE = 0.001
NUM_EPOCHS = 10

MODEL_PATH = "best_lstm.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. DISPLAY CONFIGURATION
# ============================================================

print("\n==============================")
print("LSTM TRAINING")
print("==============================")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Vocabulary size:", len(vocab))
print("Embedding dimension:", EMBEDDING_DIM)
print("Hidden dimension:", HIDDEN_DIM)
print("Bidirectional:", True)
print("Epochs:", NUM_EPOCHS)


# ============================================================
# 3. LSTM MODEL
# ============================================================

class TextLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout,
    ):
        super().__init__()

        # Convert token IDs into trainable word embeddings.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX,
        )

        # Bidirectional LSTM processes the sequence
        # from left to right and from right to left.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        # A bidirectional LSTM has two final hidden states.
        self.fc = nn.Linear(
            hidden_dim * 2,
            num_classes,
        )

    def forward(self, input_ids):

        # input_ids:
        # [batch_size, sequence_length]

        # Calculate actual sequence lengths before padding.
        lengths = input_ids.ne(PAD_IDX).sum(dim=1)

        # Avoid zero-length sequences if any empty text exists.
        lengths = lengths.clamp(min=1)

        # Embedding:
        # [batch_size, sequence_length, embedding_dim]
        embedded = self.embedding(input_ids)

        # Pack sequences so the LSTM ignores padded positions.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # h_n shape:
        # [num_layers * directions, batch_size, hidden_dim]
        _, (h_n, _) = self.lstm(packed)

        # Final forward and backward hidden states.
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]

        # Combine both directions.
        features = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1,
        )

        features = self.dropout(features)

        # Output logits:
        # [batch_size, 4]
        logits = self.fc(features)

        return logits


# ============================================================
# 4. INITIALIZE MODEL
# ============================================================

model = TextLSTM(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
).to(DEVICE)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\n==============================")
print("MODEL SUMMARY")
print("==============================")

print(model)
print(f"\nTotal parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")


# ============================================================
# 5. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 6. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for inputs, labels in loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)

        loss = criterion(logits, labels)

        loss.backward()

        # Helps prevent exploding gradients in recurrent networks.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        predictions = logits.argmax(dim=1)

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    return average_loss, accuracy


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs)

            loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)

            predictions = logits.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        all_predictions,
        all_labels,
    )


# ============================================================
# 8. TRAINING LOOP
# ============================================================

best_val_accuracy = -1.0
best_epoch = 0

print("\n==============================")
print("STARTING TRAINING")
print("==============================")

for epoch in range(1, NUM_EPOCHS + 1):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
    )

    (
        val_loss,
        val_accuracy,
        val_macro_f1,
        _,
        _,
    ) = evaluate(
        model,
        val_loader,
    )

    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    print(f"  Train Loss:      {train_loss:.4f}")
    print(f"  Train Accuracy:  {train_accuracy * 100:.2f}%")
    print(f"  Val Loss:        {val_loss:.4f}")
    print(f"  Val Accuracy:    {val_accuracy * 100:.2f}%")
    print(f"  Val Macro F1:    {val_macro_f1:.4f}")

    # Select the checkpoint using validation accuracy only.
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

        print("  Best checkpoint saved.")


# ============================================================
# 9. RESTORE BEST CHECKPOINT
# ============================================================

print("\n==============================")
print("LOADING BEST MODEL")
print("==============================")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

print("Best epoch:", best_epoch)
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


# ============================================================
# 10. FINAL TEST EVALUATION
# ============================================================

# The test set is used only after checkpoint selection.
(
    test_loss,
    test_accuracy,
    test_macro_f1,
    predictions,
    labels,
) = evaluate(
    model,
    test_loader,
)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Macro F1: {test_macro_f1:.4f}")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

class_names = [
    LABEL_NAMES[i]
    for i in range(NUM_CLASSES)
]

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    labels,
    predictions,
    labels=list(range(NUM_CLASSES)),
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print("Rows = Actual")
print("Columns = Predicted")
print()

print(
    f"{'Actual / Pred.':<16}"
    + "".join(f"{name:>12}" for name in class_names)
)

for i, row in enumerate(cm):

    print(
        f"{class_names[i]:<16}"
        + "".join(f"{value:>12}" for value in row)
    )


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("LSTM COMPLETE")
print("==============================")

print("Input: Answer / Output only")
print(f"Vocabulary size: {len(vocab):,}")
print(f"Embedding dimension: {EMBEDDING_DIM}")
print(f"Hidden dimension: {HIDDEN_DIM}")
print(f"Bidirectional: True")
print(f"Dropout: {DROPOUT}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Best epoch: {best_epoch}")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test macro F1: {test_macro_f1:.4f}")

print(f"\nModel saved to: {MODEL_PATH}")
print("\nLSTM training and evaluation complete.")