import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def find_nearest_words(word, word_vectors, word2id, id2word, top_k: int = 5):
    """
    Return the top_k nearest words to the given query word using cosine similarity.
    """
    if word not in word2id:
        return []

    query_id = word2id[word]
    query_vec = word_vectors[query_id]

    sims = []
    for idx in range(len(word_vectors)):
        if idx == query_id:
            continue
        sim = cosine_similarity(query_vec, word_vectors[idx])
        sims.append((id2word[idx], sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]