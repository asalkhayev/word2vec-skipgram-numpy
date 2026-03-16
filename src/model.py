import numpy as np


def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.W_in = rng.normal(0, 0.01, size=(vocab_size, embedding_dim))
        self.W_out = rng.normal(0, 0.01, size=(vocab_size, embedding_dim))

    def train_step(self, center_id: int, context_id: int, negative_ids: np.ndarray, learning_rate: float) -> float:
        v_c = self.W_in[center_id].copy()          # (D,)
        u_o = self.W_out[context_id].copy()        # (D,)
        u_k = self.W_out[negative_ids].copy()      # (K, D)

        pos_score = np.dot(u_o, v_c)               # scalar
        neg_scores = np.dot(u_k, v_c)              # (K,)

        pos_sig = sigmoid(pos_score)               # sigma(u_o^T v_c)
        neg_sig = sigmoid(neg_scores)              # sigma(u_k^T v_c)

        loss = -np.log(pos_sig + 1e-10) - np.sum(np.log(1.0 - neg_sig + 1e-10))

        grad_pos = pos_sig - 1.0                   # scalar
        grad_neg = neg_sig                         # (K,)

        grad_v_c = grad_pos * u_o + np.sum(grad_neg[:, None] * u_k, axis=0)   # (D,)
        grad_u_o = grad_pos * v_c                                                   # (D,)
        grad_u_k = grad_neg[:, None] * v_c                                           # (K, D)

        self.W_in[center_id] -= learning_rate * grad_v_c
        self.W_out[context_id] -= learning_rate * grad_u_o

        for i, neg_id in enumerate(negative_ids):
            self.W_out[neg_id] -= learning_rate * grad_u_k[i]

        return float(loss)

    def get_input_embeddings(self) -> np.ndarray:
        return self.W_in

    def get_output_embeddings(self) -> np.ndarray:
        return self.W_out

    def get_word_vectors(self) -> np.ndarray:
        return (self.W_in + self.W_out) / 2.0