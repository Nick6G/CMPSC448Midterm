import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = Path("TXD-22 A Large-Scale Benchmark Dataset.csv")

SOURCES = ["ChatGPT", "Grok", "Copilot", "DeepSeek"]

RANDOM_STATE = 42


# ============================================================
# 1. READ THE ORIGINAL CSV
# ============================================================

print("==============================")
print("LOADING DATASET")
print("==============================")

df = pd.read_csv(
    FILE_PATH,
    encoding="latin1"
)

print("Original dataset shape:", df.shape)
print("Original columns:", df.columns.tolist())


# ============================================================
# 2. FILTER TO THE FOUR LLMs
# ============================================================

df = df[
    df["Source"].isin(SOURCES)
].copy()

print("\n==============================")
print("FILTERED DATASET")
print("==============================")

print("Filtered dataset shape:", df.shape)

print("\nRows per source:")
print(df["Source"].value_counts())


# ============================================================
# 3. FIND QUESTIONS WITH EXACTLY ONE RESPONSE
#    FROM EACH OF THE FOUR MODELS
# ============================================================

print("\n==============================")
print("CHECKING QUESTIONS")
print("==============================")

valid_questions = []

for question, group in df.groupby("Question"):

    source_counts = group["Source"].value_counts()

    # A question is valid only if:
    #
    # 1. It has all four required sources
    # 2. Each source appears exactly once

    if (
        set(source_counts.index) == set(SOURCES)
        and all(source_counts[source] == 1 for source in SOURCES)
    ):
        valid_questions.append(question)


# Keep only the valid questions
clean_df = df[
    df["Question"].isin(valid_questions)
].copy()


print("Valid questions:", len(valid_questions))
print("Clean dataset shape:", clean_df.shape)

print("\nRows per source:")
print(clean_df["Source"].value_counts())

print("\nResponses per question:")
print(
    clean_df.groupby("Question").size().value_counts()
)


# ============================================================
# 4. VERIFY THE CLEAN DATASET
# ============================================================

expected_rows = len(valid_questions) * 4

assert len(clean_df) == expected_rows

# Every question must have exactly 4 rows
assert (
    clean_df.groupby("Question").size() == 4
).all()

# Every question must have exactly one of each source
for question, group in clean_df.groupby("Question"):

    assert set(group["Source"]) == set(SOURCES)
    assert len(group) == 4

print("\n✓ Every valid question has exactly:")
print("  - 1 ChatGPT response")
print("  - 1 Grok response")
print("  - 1 Copilot response")
print("  - 1 DeepSeek response")


# ============================================================
# 5. GET UNIQUE QUESTIONS
# ============================================================

questions = pd.Series(
    clean_df["Question"].unique()
)

print("\nTotal unique questions:", len(questions))


# ============================================================
# 6. SPLIT QUESTIONS
#
# 80% TRAIN
# 10% VALIDATION
# 10% TEST
#
# IMPORTANT:
# We split QUESTIONS, not individual rows.
# Therefore all four LLM responses to a question
# remain in the same split.
# ============================================================

train_questions, temp_questions = train_test_split(
    questions,
    test_size=0.20,
    random_state=RANDOM_STATE
)

val_questions, test_questions = train_test_split(
    temp_questions,
    test_size=0.50,
    random_state=RANDOM_STATE
)


print("\n==============================")
print("QUESTION SPLIT")
print("==============================")

print("Training questions:", len(train_questions))
print("Validation questions:", len(val_questions))
print("Test questions:", len(test_questions))

print(
    "Total questions:",
    len(train_questions)
    + len(val_questions)
    + len(test_questions)
)


# ============================================================
# 7. CREATE TRAIN / VALIDATION / TEST DATASETS
# ============================================================

train_df = clean_df[
    clean_df["Question"].isin(train_questions)
].copy()

val_df = clean_df[
    clean_df["Question"].isin(val_questions)
].copy()

test_df = clean_df[
    clean_df["Question"].isin(test_questions)
].copy()


# ============================================================
# 8. SORT EACH SPLIT
#
# This isn't required for ML, but it makes the CSVs easier
# to inspect.
# ============================================================

train_df = train_df.sort_values(
    ["Question", "Source"]
).reset_index(drop=True)

val_df = val_df.sort_values(
    ["Question", "Source"]
).reset_index(drop=True)

test_df = test_df.sort_values(
    ["Question", "Source"]
).reset_index(drop=True)


# ============================================================
# 9. VERIFY SPLIT SIZES
# ============================================================

print("\n==============================")
print("SPLIT SIZES")
print("==============================")

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

print("\nRows per source:")

print("\nTRAIN")
print(train_df["Source"].value_counts())

print("\nVALIDATION")
print(val_df["Source"].value_counts())

print("\nTEST")
print(test_df["Source"].value_counts())


# ============================================================
# 10. VERIFY NO QUESTION LEAKAGE
# ============================================================

print("\n==============================")
print("QUESTION LEAKAGE CHECK")
print("==============================")

train_question_set = set(train_df["Question"])
val_question_set = set(val_df["Question"])
test_question_set = set(test_df["Question"])

train_val_overlap = train_question_set & val_question_set
train_test_overlap = train_question_set & test_question_set
val_test_overlap = val_question_set & test_question_set

print(
    "Train ∩ Validation:",
    len(train_val_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(val_test_overlap)
)

assert len(train_val_overlap) == 0
assert len(train_test_overlap) == 0
assert len(val_test_overlap) == 0

print("\n✓ No question leakage detected.")


# ============================================================
# 11. VERIFY EACH QUESTION STILL HAS FOUR SOURCES
# ============================================================

for split_name, split_df in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    source_counts = (
        split_df
        .groupby("Question")["Source"]
        .nunique()
    )

    assert (source_counts == 4).all()

    print(
        f"✓ {split_name}: every question has all 4 LLM sources."
    )


# ============================================================
# 12. SAVE THE DATASETS
# ============================================================

train_df.to_csv(
    "train.csv",
    index=False
)

val_df.to_csv(
    "validation.csv",
    index=False
)

test_df.to_csv(
    "test.csv",
    index=False
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("FINAL DATASET SUMMARY")
print("==============================")

print(
    f"Total questions:      {len(questions)}"
)

print(
    f"Training questions:   {len(train_question_set)}"
)

print(
    f"Validation questions: {len(val_question_set)}"
)

print(
    f"Test questions:       {len(test_question_set)}"
)

print()

print(
    f"Training examples:    {len(train_df)}"
)

print(
    f"Validation examples:  {len(val_df)}"
)

print(
    f"Test examples:        {len(test_df)}"
)

print()

print("Files created:")
print("  train.csv")
print("  validation.csv")
print("  test.csv")

print("\n✓ DATA PREPARATION COMPLETE!")