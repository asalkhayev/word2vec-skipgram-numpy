# Word2Vec (Skip-Gram with Negative Sampling) – NumPy Implementation

This repository contains a minimal implementation of **Word2Vec** using the **skip-gram model with negative sampling**, implemented entirely in **NumPy** without using machine learning frameworks such as PyTorch or TensorFlow.

The goal is to demonstrate a full training loop including:

- forward pass
- loss computation
- gradient calculation
- parameter updates

---

# Model

The implementation follows the **Skip-Gram with Negative Sampling (SGNS)** approach introduced in:

Mikolov et al., 2013 — *Efficient Estimation of Word Representations in Vector Space*

Given a **center word** \(c\) and a **context word** \(o\), the model maximizes:

\[
\log \sigma(u_o^T v_c)
\]

while minimizing similarity with sampled **negative words** \(n_i\):

\[
\sum_i \log \sigma(-u_{n_i}^T v_c)
\]

Where:

- \(v_c\) = input embedding of center word  
- \(u_o\) = output embedding of context word

Total loss:

\[
L = -\log \sigma(u_o^T v_c) - \sum_i \log \sigma(-u_{n_i}^T v_c)
\]

---

# Training Procedure

1. Preprocess text corpus
2. Build vocabulary
3. Generate skip-gram training pairs
4. Sample negative words
5. Compute forward pass
6. Compute gradients
7. Update embeddings using SGD

---

# Project Structure
```
word2vec-numpy
│
├── data/
│   └── sample.txt
│
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```
---

# Running the Code

Create a Python environment and install dependencies:

```
pip install -r requirements.txt
```

---

Run training:

```
python src/train.py
```

The script will:

- train word embeddings
- print training loss
- show nearest neighbors for example words

---

# Example Output
```
Epoch 1/50, Avg Loss: 4.12
Epoch 10/50, Avg Loss: 3.78
Epoch 20/50, Avg Loss: 3.51

Nearest neighbors:
alice: sister book mind thought
book: pictures reading sister
```

---

# Possible Improvements

Several extensions could improve this implementation:

- subsampling of frequent words
- dynamic context windows
- mini-batch training
- hierarchical softmax
- optimized negative sampling

---

# Implementation Notes

This project focuses on clarity rather than efficiency.  
All gradients and updates are implemented manually in NumPy to clearly demonstrate the underlying learning algorithm.
