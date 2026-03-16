import re
from collections import Counter


def tokenize(text: str):
    """
    Lowercase text and keep only alphabetic tokens and apostrophes.
    """
    text = text.lower()
    return re.findall(r"[a-z']+", text)


def preprocess_corpus(file_path: str, min_count: int = 1):
    """
    Read corpus, tokenize it, build vocabulary, and convert corpus to ids.

    Returns a dictionary containing:
    - tokens
    - filtered_tokens
    - word_counts
    - word2id
    - id2word
    - corpus_ids
    - vocab_size
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenize(text)
    word_counts = Counter(tokens)

    filtered_tokens = [word for word in tokens if word_counts[word] >= min_count]

    vocab = sorted(set(filtered_tokens))
    word2id = {word: i for i, word in enumerate(vocab)}
    id2word = {i: word for word, i in word2id.items()}

    corpus_ids = [word2id[word] for word in filtered_tokens]

    filtered_word_counts = Counter(filtered_tokens)

    return {
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "word_counts": filtered_word_counts,
        "word2id": word2id,
        "id2word": id2word,
        "corpus_ids": corpus_ids,
        "vocab_size": len(vocab),
    }