import numpy as np


def generate_skipgram_pairs(corpus_ids: list[int], window_size: int) -> list[tuple[int, int]]:
    pairs = []
    n = len(corpus_ids)

    for center_pos in range(n):
        center_word = corpus_ids[center_pos]

        left = max(0, center_pos - window_size)
        right = min(n, center_pos + window_size + 1)

        for context_pos in range(left, right):
            if context_pos == center_pos:
                continue
            context_word = corpus_ids[context_pos]
            pairs.append((center_word, context_word))

    return pairs


def build_negative_sampling_distribution(word_counts, word2id):
    vocab_size = len(word2id)
    freqs = np.zeros(vocab_size, dtype=np.float64)

    for word, count in word_counts.items():
        freqs[word2id[word]] = count

    freqs = freqs ** 0.75
    probs = freqs / freqs.sum()

    return probs


def sample_negative_words(
    probs: np.ndarray,
    num_negatives: int,
    positive_id: int,
) -> np.ndarray:
    negatives = []
    vocab_size = len(probs)

    while len(negatives) < num_negatives:
        sample = np.random.choice(vocab_size, p=probs)
        if sample != positive_id:
            negatives.append(sample)

    return np.array(negatives, dtype=np.int64)


if __name__ == "__main__":
    from preprocess import preprocess_corpus

    data = preprocess_corpus("data/sample.txt", min_count=1)

    pairs = generate_skipgram_pairs(data["corpus_ids"], window_size=2)
    probs = build_negative_sampling_distribution(data["word_counts"], data["word2id"])

    print("Number of training pairs:", len(pairs))
    print("First 10 pairs:", pairs[:10])

    center_id, context_id = pairs[0]
    negatives = sample_negative_words(probs, num_negatives=5, positive_id=context_id)

    print("Example pair:", (center_id, context_id))
    print("Negative samples:", negatives)