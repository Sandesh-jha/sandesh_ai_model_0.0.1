# Sandesh AI Model v0.0.1

A 15.7M parameter GPT-style Transformer trained from scratch using PyTorch.

---

## 📥 How To Download This Project

### Option 1 — Download ZIP (Recommended)

1. Open this repository page.
2. Click the green **Code** button.
3. Click **Download ZIP**.
4. Extract the ZIP file.
5. Open the extracted folder.

### Option 2 — Clone Using Git

```bash
git clone https://github.com/YOUR_USERNAME/sandesh_ai_model_0.0.1.git
```

```bash
cd sandesh_ai_model_0.0.1
```

---

## 🖥 System Requirements

You must install:

- Python 3.10 or newer
- pip (comes with Python)

---

## 🔹 Step 1 — Install Python

1. Go to: https://www.python.org/downloads/
2. Download Python 3.10 or newer.
3. During installation, check:
   ✔ "Add Python to PATH"
4. Finish installation.

Verify installation:

```bash
python --version
```

---

## 🔹 Step 2 — Install Required Libraries

Open terminal inside the project folder and run:

```bash
pip install torch tokenizers
```

If pip does not work, try:

```bash
python -m pip install torch tokenizers
```

---

## 🔹 Step 3 — Run The Model

Inside the project folder, run:

```bash
python run.py
```

You will see:

```
You:
```

Type your prompt. Example:

```
The Earth
```

To exit the program:

```
exit
```

---

## 📁 Project Files

- model.py → Transformer architecture  
- run.py → Text generation script  
- tokenizer.json → Tokenizer file  
- sandesh_ai_m_001.pt → Trained model weights (~60MB)  

---

## ⚠ Notes

- This is a small experimental model.
- CPU is enough to run it.
- It may repeat or memorize training data.
- Performance depends on your system speed.

---

## 📜 License

MIT License
