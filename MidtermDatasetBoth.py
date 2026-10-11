import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CONFIGURATION
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

# Maximum tokens per example
MAX_LENGTH = 512

# Batch size
BATCH_SIZE = 64

# Reproducibility
SEED = 42

torch.manual_seed(SEED)


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Normalize text before tokenization.
    """

    if pd.isna(text):
        text = ""
    else:
        text = str(text)

    # Fix common encoding artifacts.
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.lower()

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 3. TOKENIZER
# ============================================================

def tokenize(text):
    """
    Split text into words, numbers, contractions,
    and punctuation.
    """

    text = clean_text(text)

    tokens = re.findall(
        r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?|[^\w\s]",
        text,
    )

    return tokens


# ============================================================
# 4. VOCABULARY
# ============================================================

class Vocabulary:

    def __init__(self, max_size=MAX_VOCAB_SIZE):

        self.max_size = max_size

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
            counter.update(tokenize(text))

        # Reserve two slots for PAD and UNK.
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

        ids = [
            self.word_to_idx.get(token, UNK_IDX)
            for token in tokens
        ]

        return ids

    def __len__(self):
        return len(self.word_to_idx)


# ============================================================
# 5. PYTORCH DATASET -- QUESTION + ANSWER
# ============================================================

class LLMDataset(Dataset):

    def __init__(
        self,
        dataframe,
        vocabulary,
        max_length=MAX_LENGTH,
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.vocabulary = vocabulary
        self.max_length = max_length

        # Check that the required columns exist.
        required_columns = {
            "Question",
            "Answer",
            "Source",
        }

        missing_columns = (
            required_columns - set(self.dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # Check that all labels are recognized.
        unknown_sources = (
            set(self.dataframe["Source"].dropna().unique())
            - set(LABEL_MAP.keys())
        )

        if unknown_sources:
            raise ValueError(
                f"Unknown LLM sources: {unknown_sources}"
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        # ----------------------------------------------------
        # Get question and answer
        # ----------------------------------------------------

        question = row["Question"]
        answer = row["Answer"]

        if pd.isna(question):
            question = ""
        else:
            question = str(question)

        if pd.isna(answer):
            answer = ""
        else:
            answer = str(answer)

        # ----------------------------------------------------
        # BOTH INPUTS:
        # Question + separator + Answer
        # ----------------------------------------------------

        text = question + " [SEP] " + answer

        # ----------------------------------------------------
        # Convert text into token IDs
        # ----------------------------------------------------

        token_ids = self.vocabulary.encode(text)

        # Truncate long sequences.
        token_ids = token_ids[:self.max_length]

        # Pad shorter sequences.
        if len(token_ids) < self.max_length:
            token_ids += [PAD_IDX] * (
                self.max_length - len(token_ids)
            )

        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Target label: LLM that generated the answer
        # ----------------------------------------------------

        label = torch.tensor(
            LABEL_MAP[row["Source"]],
            dtype=torch.long,
        )

        return input_ids, label


# ============================================================
# 6. LOAD DATA
# ============================================================

print("==============================")
print("LOADING INPUT + OUTPUT DATA")
print("==============================")

train_df = pd.read_csv(
    TRAIN_FILE,
    encoding="latin1",
)

val_df = pd.read_csv(
    VAL_FILE,
    encoding="latin1",
)

test_df = pd.read_csv(
    TEST_FILE,
    encoding="latin1",
)

print("Training examples:", len(train_df))
print("Validation examples:", len(val_df))
print("Test examples:", len(test_df))


# ============================================================
# 7. VALIDATE DATA
# ============================================================

print("\n==============================")
print("VALIDATING DATA")
print("==============================")

for name, dataframe in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:

    required_columns = {
        "Question",
        "Answer",
        "Source",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: {missing_columns}"
        )

    if dataframe["Question"].isna().any():
        raise ValueError(
            f"{name} contains missing questions."
        )

    if dataframe["Answer"].isna().any():
        raise ValueError(
            f"{name} contains missing answers."
        )

    if dataframe["Source"].isna().any():
        raise ValueError(
            f"{name} contains missing source labels."
        )

    unknown_sources = (
        set(dataframe["Source"].unique())
        - set(LABEL_MAP.keys())
    )

    if unknown_sources:
        raise ValueError(
            f"{name} contains unknown sources: {unknown_sources}"
        )

    print(f"{name}: {len(dataframe)} rows verified.")


# ============================================================
# 8. BUILD VOCABULARY FROM TRAINING DATA ONLY
# ============================================================

print("\n==============================")
print("BUILDING QUESTION + ANSWER VOCABULARY")
print("==============================")

# Both questions and answers are included.
# Validation and test text are NOT used to build the vocabulary.

training_texts = []

for _, row in train_df.iterrows():

    question = str(row["Question"])
    answer = str(row["Answer"])

    combined_text = question + " [SEP] " + answer

    training_texts.append(combined_text)

vocab = Vocabulary(
    max_size=MAX_VOCAB_SIZE,
)

vocab.build(training_texts)


# ============================================================
# 9. CREATE DATASETS
# ============================================================

print("\n==============================")
print("CREATING INPUT + OUTPUT DATASETS")
print("==============================")

train_dataset = LLMDataset(
    train_df,
    vocab,
    max_length=MAX_LENGTH,
)

val_dataset = LLMDataset(
    val_df,
    vocab,
    max_length=MAX_LENGTH,
)

test_dataset = LLMDataset(
    test_df,
    vocab,
    max_length=MAX_LENGTH,
)

print("Mode: input_output (Question + Answer)")
print("Training dataset:", len(train_dataset))
print("Validation dataset:", len(val_dataset))
print("Test dataset:", len(test_dataset))


# ============================================================
# 10. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# 11. TEST THE DATALOADER
# ============================================================

print("\n==============================")
print("TESTING DATALOADER")
print("==============================")

inputs, labels = next(iter(train_loader))

print("Input tensor shape:", inputs.shape)
print("Label tensor shape:", labels.shape)
print("Input tensor dtype:", inputs.dtype)
print("Label tensor dtype:", labels.dtype)

print(
    "First example token IDs:",
    inputs[0][:20].tolist(),
)

print("First example label:", labels[0].item())

print(
    "First example LLM:",
    LABEL_NAMES[labels[0].item()],
)


# ============================================================
# 12. VERIFY CLASS BALANCE
# ============================================================

print("\n==============================")
print("LABEL DISTRIBUTION")
print("==============================")

for name, dataframe in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:

    counts = dataframe["Source"].value_counts()

    print(f"\n{name}:")

    for source in LABEL_MAP:
        print(
            f"  {source}: {int(counts.get(source, 0))}"
        )

    expected_counts = [
        int(counts.get(source, 0))
        for source in LABEL_MAP
    ]

    if len(set(expected_counts)) != 1:
        raise ValueError(
            f"{name} is not balanced across all four sources."
        )


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("INPUT + OUTPUT PIPELINE READY")
print("==============================")

print("Input: Question + Answer")
print("Target: LLM source")
print(f"Vocabulary size: {len(vocab):,}")
print(f"Maximum sequence length: {MAX_LENGTH}")
print(f"Batch size: {BATCH_SIZE}")

print("\n✓ Questions and answers included")
print("✓ Separator included between question and answer")
print("✓ Vocabulary built from training data only")
print("✓ Existing train/validation/test splits preserved")
print("✓ Four balanced LLM classes")
print("✓ PyTorch Dataset working")
print("✓ DataLoaders working")