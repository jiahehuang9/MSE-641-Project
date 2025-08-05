import re
import unicodedata
import spacy
from nltk.stem import SnowballStemmer
from datasets import load_dataset
from typing import List
import multiprocessing

# Initialization
nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer("english")

# Text Cleaning Functions
def clean_text(text: str) -> str:
    # 1. Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])

    # 2. Lowercasing
    text = text.lower()

    # 3. Remove URLs and emails
    text = re.sub(r'https?://\S+|www\.\S+|\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)

    # 4. Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # 5. Replace all numbers with "NUM"
    text = re.sub(r'\d+', ' NUM ', text)

    # 6. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def advanced_process(text: str, remove_stop=True, do_lemma=True, do_stem=True) -> str:
    doc = nlp(text)
    tokens: List[str] = []

    for token in doc:
        if remove_stop and token.is_stop:
            continue
        if token.is_punct or not token.text.strip():
            continue

        word = token.lemma_ if do_lemma else token.text
        word = word.lower()

        if do_stem:
            word = stemmer.stem(word)

        tokens.append(word)

    return " ".join(tokens)


def clean_sample(example):
    if "conversation" in example:
        example["conversation"] = [
            {
                **msg,
                "content_clean": advanced_process(
                    clean_text(msg["content"]),
                    remove_stop=True,
                    do_lemma=True,
                    do_stem=True
                ) if "content" in msg else ""
            }
            for msg in example["conversation"]
        ]
    return example


# Main Processing
if __name__ == '__main__':
    print("Loading dataset...")
    ds = load_dataset("allenai/WildChat", split="train")

    print("Filtering English conversations...")
    ds_en = ds.filter(lambda x: x["language"] == "English")

    print("Cleaning conversations with multi-processing...")
    ds_en_cleaned = ds_en.map(clean_sample, num_proc=12)

    print("Saving cleaned dataset to JSONL...")
    ds_en_cleaned.to_json("wildchat_en_cleaned.jsonl", orient="records", lines=True)
    print("Done: saved to 'wildchat_en_cleaned.jsonl'")
