import torch
import torch.nn as nn
import torch.optim as optim

from MidtermDatasetBoth import (
    train_loader,
    val_loader,
    test_loader,
    vocab,
    LABEL_NAMES,
    PAD_IDX,
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 4

EMBEDDING_DIM = 128

NUM_FILTERS = 128

KERNEL_SIZES = [3, 4, 5]

DROPOUT = 0.5

LEARNING_RATE = 0.001

NUM_EPOCHS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PRINT DEVICE
# ============================================================

print("==============================")
print("CNN TRAINING")
print("==============================")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# CNN MODEL
# ============================================================

class TextCNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_filters,
        kernel_sizes,
        num_classes,
        dropout
    ):

        super().__init__()

        # ----------------------------------------------------
        # Embedding layer
        # ----------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX
        )

        # ----------------------------------------------------
        # Convolution layers
        #
        # Each convolution looks at a different number
        # of neighboring words.
        #
        # Kernel 3 -> 3-word patterns
        # Kernel 4 -> 4-word patterns
        # Kernel 5 -> 5-word patterns
        # ----------------------------------------------------

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=kernel_size
                )
                for kernel_size in kernel_sizes
            ]
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(dropout)

        # ----------------------------------------------------
        # Final classification layer
        # ----------------------------------------------------

        self.fc = nn.Linear(
            num_filters * len(kernel_sizes),
            num_classes
        )

    def forward(self, x):

        # x shape:
        # [batch_size, sequence_length]

        # ----------------------------------------------------
        # Convert token IDs to embeddings
        # ----------------------------------------------------

        x = self.embedding(x)

        # Shape:
        # [batch_size, sequence_length, embedding_dim]

        # Conv1D expects:
        # [batch_size, channels, sequence_length]

        x = x.permute(0, 2, 1)

        # ----------------------------------------------------
        # Apply convolution + ReLU + max pooling
        # ----------------------------------------------------

        pooled_outputs = []

        for conv in self.convs:

            # Convolution
            feature_map = conv(x)

            # ReLU activation
            feature_map = torch.relu(feature_map)

            # Global max pooling
            pooled = torch.max(
                feature_map,
                dim=2
            ).values

            pooled_outputs.append(pooled)

        # ----------------------------------------------------
        # Combine features from all kernel sizes
        # ----------------------------------------------------

        x = torch.cat(
            pooled_outputs,
            dim=1
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        x = self.dropout(x)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        output = self.fc(x)

        return output


# ============================================================
# CREATE MODEL
# ============================================================

model = TextCNN(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    num_filters=NUM_FILTERS,
    kernel_sizes=KERNEL_SIZES,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)

model = model.to(DEVICE)


# ============================================================
# DISPLAY MODEL
# ============================================================

print("\n==============================")
print("MODEL")
print("==============================")

print(model)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "\nTotal parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)


# ============================================================
# LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    total_loss = 0.0

    correct = 0
    total = 0

    for inputs, labels in loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Track statistics
        total_loss += (
            loss.item() * inputs.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = total_loss / total

    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0.0

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)

            # Loss
            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * inputs.size(0)
            )

            # Predictions
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    average_loss = total_loss / total

    accuracy = correct / total

    return (
        average_loss,
        accuracy,
        all_predictions,
        all_labels
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_accuracy = 0.0

best_epoch = 0

print("\n==============================")
print("STARTING TRAINING")
print("==============================")


for epoch in range(1, NUM_EPOCHS + 1):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader
    )

    (
        val_loss,
        val_accuracy,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )

    print(
        f"\nEpoch {epoch}/{NUM_EPOCHS}"
    )

    print(
        f"  Train Loss: {train_loss:.4f}"
    )

    print(
        f"  Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"  Val Loss: {val_loss:.4f}"
    )

    print(
        f"  Val Accuracy: {val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Save the best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_epoch = epoch

        torch.save(
            model.state_dict(),
            "best_cnn.pt"
        )

        print(
            "  ✓ Best model saved."
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n==============================")
print("LOADING BEST MODEL")
print("==============================")

model.load_state_dict(
    torch.load(
        "best_cnn.pt",
        map_location=DEVICE
    )
)

print(
    "Best epoch:",
    best_epoch
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

(
    test_loss,
    test_accuracy,
    predictions,
    labels
) = evaluate(
    model,
    test_loader
)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================================
# CLASS-BY-CLASS RESULTS
# ============================================================

print("\n==============================")
print("CLASS ACCURACY")
print("==============================")

for class_id in range(NUM_CLASSES):

    class_predictions = [
        prediction
        for prediction, label
        in zip(predictions, labels)
        if label == class_id
    ]

    class_labels = [
        label
        for label in labels
        if label == class_id
    ]

    if len(class_labels) > 0:

        class_correct = sum(
            prediction == label
            for prediction, label
            in zip(
                class_predictions,
                class_labels
            )
        )

        class_accuracy = (
            class_correct /
            len(class_labels)
        )

        print(
            f"{LABEL_NAMES[class_id]}: "
            f"{class_accuracy * 100:.2f}% "
            f"({class_correct}/{len(class_labels)})"
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    labels,
    predictions
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    "             "
    + " ".join(
        f"{LABEL_NAMES[i]:>10}"
        for i in range(NUM_CLASSES)
    )
)

for i, row in enumerate(cm):

    print(
        f"{LABEL_NAMES[i]:>12}"
        + " ".join(
            f"{value:>10}"
            for value in row
        )
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==============================")
print("CNN COMPLETE")
print("==============================")

print(
    "Input: Answer / Output only"
)

print(
    f"Vocabulary: {len(vocab):,}"
)

print(
    f"Embedding dimension: {EMBEDDING_DIM}"
)

print(
    f"Filters per kernel: {NUM_FILTERS}"
)

print(
    f"Kernel sizes: {KERNEL_SIZES}"
)

print(
    f"Dropout: {DROPOUT}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Epochs: {NUM_EPOCHS}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Final test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\n✓ CNN training and evaluation complete."
)