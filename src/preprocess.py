import re
from collections import Counter


def read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", text)
    return tokens


def build_vocab(tokens: list[str], min_count: int = 1) -> tuple[dict[str, int], dict[int, str], Counter]:
    counter = Counter(tokens)

    vocab_words = [word for word, count in counter.items() if count >= min_count]
    vocab_words.sort()

    word2id = {word: i for i, word in enumerate(vocab_words)}
    id2word = {i: word for word, i in word2id.items()}

    filtered_counter = Counter({word: counter[word] for word in vocab_words})

    return word2id, id2word, filtered_counter


def text_to_ids(tokens: list[str], word2id: dict[str, int]) -> list[int]:
    return [word2id[word] for word in tokens if word in word2id]


def preprocess_corpus(file_path: str, min_count: int = 1):
    text = read_text(file_path)
    tokens = tokenize(text)
    word2id, id2word, word_counts = build_vocab(tokens, min_count=min_count)
    corpus_ids = text_to_ids(tokens, word2id)

    return {
        "text": text,
        "tokens": tokens,
        "word2id": word2id,
        "id2word": id2word,
        "word_counts": word_counts,
        "corpus_ids": corpus_ids,
        "vocab_size": len(word2id),
    }


if __name__ == "__main__":
    data = preprocess_corpus("data/sample.txt", min_count=1)

    print("Vocab size:", data["vocab_size"])
    print("First 20 tokens:", data["tokens"][:20])
    print("First 20 ids:", data["corpus_ids"][:20])