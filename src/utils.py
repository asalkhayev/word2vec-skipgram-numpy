import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def find_nearest_words(
    word: str,
    word_vectors: np.ndarray,
    word2id: dict[str, int],
    id2word: dict[int, str],
    top_k: int = 5,
) -> list[str]:
    if word not in word2id:
        return []

    word_id = word2id[word]
    query_vector = word_vectors[word_id]

    similarities = []
    for i in range(len(word_vectors)):
        if i == word_id:
            continue

        sim = cosine_similarity(query_vector, word_vectors[i])
        similarities.append((sim, id2word[i]))

    similarities.sort(reverse=True, key=lambda x: x[0])

    return [word for _, word in similarities[:top_k]]