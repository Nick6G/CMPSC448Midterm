import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "train.csv"
VAL_FILE = "validation.csv"
TEST_FILE = "test.csv"

# Four LLM classes
LABEL_MAP = {
    "ChatGPT": 0,
    "Grok": 1,
    "Copilot": 2,
    "DeepSeek": 3,
}

LABEL_NAMES = {
    0: "ChatGPT",
    1: "Grok",
    2: "Copilot",
    3: "DeepSeek",
}

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
UNK_IDX = 1

# Vocabulary settings
MAX_VOCAB_SIZE = 30000

# Maximum number of tokens per example
MAX_LENGTH = 512


# ============================================================
# 1. TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean and normalize text before tokenization.

    This function:
    - Converts the input to a string
    - Fixes common mojibake characters caused by encoding issues
    - Converts text to lowercase
    - Removes unnecessary whitespace
    """

    text = str(text)

    # Fix common encoding artifacts found in the dataset.
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "â€™": "'",
        "â": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


# ============================================================
# 2. TOKENIZER
# ============================================================

def tokenize(text):
    """
    Convert text into a list of word/punctuation tokens.

    Example:

        "What is machine learning?"

    becomes approximately:

        ["what", "is", "machine", "learning", "?"]
    """

    text = clean_text(text)

    # Keep words, numbers, contractions, and punctuation.
    tokens = re.findall(
        r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?|[^\w\s]",
        text
    )

    return tokens


# ============================================================
# 3. VOCABULARY
# ============================================================

class Vocabulary:

    def __init__(self, max_size=MAX_VOCAB_SIZE):

        self.max_size = max_size

        # Start with special tokens.
        self.word_to_idx = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
        }

        self.idx_to_word = {
            PAD_IDX: PAD_TOKEN,
            UNK_IDX: UNK_TOKEN,
        }

    def build(self, texts):

        counter = Counter()

        for text in texts:

            tokens = tokenize(text)

            counter.update(tokens)

        # Most common words first
        most_common = counter.most_common(
            self.max_size - 2
        )

        for word, _ in most_common:

            if word not in self.word_to_idx:

                idx = len(self.word_to_idx)

                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word

        print(
            f"Vocabulary size: {len(self.word_to_idx):,}"
        )

    def encode(self, text):

        tokens = tokenize(text)

        ids = []

        for token in tokens:

            if token in self.word_to_idx:

                ids.append(
                    self.word_to_idx[token]
                )

            else:

                ids.append(UNK_IDX)

        return ids

    def __len__(self):

        return len(self.word_to_idx)


# ============================================================
# 4. PYTORCH DATASET
# ============================================================

class LLMDataset(Dataset):

    def __init__(
        self,
        dataframe,
        vocabulary,
        mode="output",
        max_length=MAX_LENGTH
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.vocabulary = vocabulary

        self.mode = mode

        self.max_length = max_length

        # Verify mode
        valid_modes = [
            "output",
            "input",
            "input_output"
        ]

        if self.mode not in valid_modes:

            raise ValueError(
                f"mode must be one of {valid_modes}"
            )

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        question = str(row["Question"])
        answer = str(row["Answer"])

        # ----------------------------------------------------
        # Select which text the model receives
        # ----------------------------------------------------

        if self.mode == "output":

            text = answer

        elif self.mode == "input":

            text = question

        elif self.mode == "input_output":

            text = question + " [SEP] " + answer

        # ----------------------------------------------------
        # Convert text to token IDs
        # ----------------------------------------------------

        token_ids = self.vocabulary.encode(text)

        # ----------------------------------------------------
        # Truncate long sequences
        # ----------------------------------------------------

        token_ids = token_ids[:self.max_length]

        # ----------------------------------------------------
        # Pad short sequences
        # ----------------------------------------------------

        if len(token_ids) < self.max_length:

            token_ids += [
                PAD_IDX
            ] * (
                self.max_length - len(token_ids)
            )

        # Convert to PyTorch tensor
        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long
        )

        # Convert LLM name to numerical label
        label = LABEL_MAP[row["Source"]]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return input_ids, label


# ============================================================
# 5. LOAD DATA
# ============================================================

print("==============================")
print("LOADING DATA")
print("==============================")

train_df = pd.read_csv(
    TRAIN_FILE,
    encoding="latin1"
)

val_df = pd.read_csv(
    VAL_FILE,
    encoding="latin1"
)

test_df = pd.read_csv(
    TEST_FILE,
    encoding="latin1"
)

print("Training examples:", len(train_df))
print("Validation examples:", len(val_df))
print("Test examples:", len(test_df))


# ============================================================
# 6. BUILD VOCABULARY USING TRAINING DATA ONLY
# ============================================================

print("\n==============================")
print("BUILDING VOCABULARY")
print("==============================")

# IMPORTANT:
# The vocabulary is built ONLY from training data.
#
# This prevents information from validation/test data
# from leaking into the training process.

training_texts = []

for _, row in train_df.iterrows():

    training_texts.append(
        str(row["Question"])
    )

    training_texts.append(
        str(row["Answer"])
    )


vocab = Vocabulary(
    max_size=MAX_VOCAB_SIZE
)

vocab.build(training_texts)


# ============================================================
# 7. CREATE DATASETS
# ============================================================

# Default experiment:
# OUTPUT ONLY
#
# We will change this to "input" or "input_output"
# for RQ2 later.

MODE = "output"


print("\n==============================")
print("CREATING DATASETS")
print("==============================")

train_dataset = LLMDataset(
    train_df,
    vocab,
    mode=MODE,
    max_length=MAX_LENGTH
)

val_dataset = LLMDataset(
    val_df,
    vocab,
    mode=MODE,
    max_length=MAX_LENGTH
)

test_dataset = LLMDataset(
    test_df,
    vocab,
    mode=MODE,
    max_length=MAX_LENGTH
)

print("Mode:", MODE)

print(
    "Training dataset:",
    len(train_dataset)
)

print(
    "Validation dataset:",
    len(val_dataset)
)

print(
    "Test dataset:",
    len(test_dataset)
)


# ============================================================
# 8. CREATE DATALOADERS
# ============================================================

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 9. TEST THE DATALOADER
# ============================================================

print("\n==============================")
print("TESTING DATALOADER")
print("==============================")

inputs, labels = next(
    iter(train_loader)
)

print("Input tensor shape:", inputs.shape)
print("Label tensor shape:", labels.shape)

print(
    "Input tensor dtype:",
    inputs.dtype
)

print(
    "Label tensor dtype:",
    labels.dtype
)

print(
    "First example token IDs:",
    inputs[0][:20].tolist()
)

print(
    "First example label:",
    labels[0].item()
)

print(
    "First example LLM:",
    LABEL_NAMES[labels[0].item()]
)


# ============================================================
# 10. CHECK LABEL DISTRIBUTION
# ============================================================

print("\n==============================")
print("LABEL DISTRIBUTION")
print("==============================")

for name, loader in [
    ("Train", train_loader),
    ("Validation", val_loader),
    ("Test", test_loader)
]:

    counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    for _, labels in loader:

        for label in labels.tolist():

            counts[label] += 1

    print(f"\n{name}:")

    for label, count in counts.items():

        print(
            f"  {LABEL_NAMES[label]}: {count}"
        )


# ============================================================
# 11. FINAL CHECK
# ============================================================

print("\n==============================")
print("DATA PIPELINE READY")
print("==============================")

print(
    f"Vocabulary size: {len(vocab):,}"
)

print(
    f"Maximum sequence length: {MAX_LENGTH}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Input mode: {MODE}"
)

print("\n✓ Tokenizer working")
print("✓ Vocabulary working")
print("✓ PyTorch Dataset working")
print("✓ DataLoader working")
print("✓ Labels working")