import numpy as np

from preprocess import preprocess_corpus
from dataset import (
    generate_skipgram_pairs,
    build_negative_sampling_distribution,
    sample_negative_words,
)
from model import SkipGramNegativeSampling
from utils import find_nearest_words


def main():
    np.random.seed(42)

    data = preprocess_corpus("data/sample.txt", min_count=1)

    print("Vocab size:", data["vocab_size"])
    print("Corpus length:", len(data["corpus_ids"]))

    pairs = generate_skipgram_pairs(data["corpus_ids"], window_size=2)
    print("Number of training pairs:", len(pairs))

    if len(pairs) == 0:
        print("No training pairs found. Put more text into data/sample.txt")
        return

    probs = build_negative_sampling_distribution(
        data["word_counts"],
        data["word2id"],
    )

    model = SkipGramNegativeSampling(
        vocab_size=data["vocab_size"],
        embedding_dim=30,
        seed=42,
    )

    learning_rate = 0.05
    num_negatives = 5
    epochs = 50

    loss_history = []

    for epoch in range(epochs):
        np.random.shuffle(pairs)
        total_loss = 0.0

        for center_id, context_id in pairs:
            negative_ids = sample_negative_words(
                probs=probs,
                num_negatives=num_negatives,
                forbidden_id=context_id,
            )

            loss = model.train_step(
                center_id=center_id,
                context_id=context_id,
                negative_ids=negative_ids,
                learning_rate=learning_rate,
            )
            total_loss += loss

        avg_loss = total_loss / len(pairs)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    word_vectors = model.get_word_vectors()

    test_words = ["alice", "book", "sister", "pictures"]
    print("\nNearest neighbors:")
    for word in test_words:
        if word in data["word2id"]:
            neighbors = find_nearest_words(
                word=word,
                word_vectors=word_vectors,
                word2id=data["word2id"],
                id2word=data["id2word"],
                top_k=5,
            )
            print(f"{word}: {neighbors}")


if __name__ == "__main__":
    main()