import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / 'youtube_comments_40k.csv'
OUTPUT_FILE = BASE_DIR / 'youtube_comments_40k_cleaned.csv'

# ── 1. Load the raw data ──────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
print(f"Original rows: {len(df)}")

# ── 2. Drop nulls ─────────────────────────────────────────────────────────────
# Some comments have no text at all (deleted comments, API gaps)
df = df.dropna(subset=['text'])
print(f"After dropping null text: {len(df)}")

# ── 3. Drop duplicate comments ────────────────────────────────────────────────
# Same author posting the exact same comment (spam / copy-paste bots)
df = df.drop_duplicates(subset=['author', 'text'])
print(f"After dropping duplicates: {len(df)}")

# ── 4. Fix the date column ────────────────────────────────────────────────────
# Right now updated_at is just a string like "2026-02-25T00:49:20Z"
# We convert it to a real datetime so we can sort and filter by date later
df['updated_at'] = pd.to_datetime(df['updated_at'], utc=True)

# ── 5. Clean the text ─────────────────────────────────────────────────────────
def clean_text(text):
    # a) Remove URLs (http://... or https://...)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # b) Remove HTML tags like <br>, <b>, etc.
    text = re.sub(r'<.*?>', '', text)
    # c) Fix HTML entities like &amp; → & and &#39; → '
    text = text.replace('&amp;', '&').replace('&#39;', "'").replace('&quot;', '"')
    # d) Remove excessive whitespace / newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text_clean'] = df['text'].apply(clean_text)

# ── 6. Drop very short comments ───────────────────────────────────────────────
# Comments like "👍", "lol", "yes" carry no useful meaning for NLP
# We keep comments with at least 4 characters after cleaning
df = df[df['text_clean'].str.len() >= 4]
print(f"After dropping very short comments: {len(df)}")

# ── 7. Reset the index ────────────────────────────────────────────────────────
df = df.reset_index(drop=True)

# ── 8. Save the cleaned dataset ───────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nFinal cleaned dataset: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print("\nSample cleaned text:")
print(df[['text', 'text_clean']].head(3).to_string())
