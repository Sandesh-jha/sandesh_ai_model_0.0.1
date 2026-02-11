# 🧠 Sandesh AI Model v0.0.1
A custom 15.7M parameter GPT-style Transformer trained completely from scratch using PyTorch.

---

## 🚀 Model Overview

- Architecture: Decoder-only Transformer (GPT-style)
- Layers: 6
- Embedding Dimension: 384
- Attention Heads: 6
- Feedforward Size: 1536
- Context Length: 128 tokens
- Vocabulary Size: 6592
- Total Parameters: 15,765,952
- Model Size: ~60MB

---

## 📊 Training Details

- Dataset size: 3535 text samples
- Total tokens: ~89,000
- Epochs trained: 50
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Gradient Clipping: Enabled

---

## 📦 Installation

```bash
pip install torch tokenizers
