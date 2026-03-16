import numpy as np


def generate_skipgram_pairs(corpus_ids, window_size: int = 2):
    """
    Generate (center, context) pairs for skip-gram training.
    """
    pairs = []

    for i, center_id in enumerate(corpus_ids):
        left = max(0, i - window_size)
        right = min(len(corpus_ids), i + window_size + 1)

        for j in range(left, right):
            if i != j:
                context_id = corpus_ids[j]
                pairs.append((center_id, context_id))

    return pairs


def build_negative_sampling_distribution(word_counts, word2id):
    """
    Build the standard word2vec negative sampling distribution:
    P(w) proportional to count(w)^0.75
    """
    vocab_size = len(word2id)
    freqs = np.zeros(vocab_size, dtype=np.float64)

    for word, idx in word2id.items():
        freqs[idx] = word_counts[word]

    probs = freqs ** 0.75
    probs /= probs.sum()
    return probs


def sample_negative_words(probs, num_negatives: int, forbidden_id: int):
    """
    Sample negative word ids, excluding the true context word.
    """
    negatives = []

    while len(negatives) < num_negatives:
        candidate = np.random.choice(len(probs), p=probs)
        if candidate != forbidden_id:
            negatives.append(candidate)

    return np.array(negatives, dtype=np.int64)