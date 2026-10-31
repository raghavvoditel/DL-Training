# DL Training

My deep learning training projects, from a first Streamlit app to a tiny GPT built from scratch.
Every project is kept small and simple, so the whole idea fits in one or two files.

## Projects

| # | Folder | What it does | Main tools |
|---|---|---|---|
| 01 | `01_streamlit_intro` | Marks calculator web app: grade, table and bar chart | Streamlit, pandas |
| 02 | `02_ollama_chatbot` | Local chatbot that remembers the conversation | Ollama (llama3.2:1b), Streamlit |
| 03 | `03_cpp_notes_rag` | Ask questions about my C++ notes (RAG in the terminal) | Ollama embeddings, NumPy |
| 04 | `04_pdf_qa_app` | Upload a PDF/TXT and chat with it | pypdf, Ollama, Streamlit |
| 05 | `05_college_helpdesk` | College FAQ bot with sources + automatic tests | TF-IDF (scikit-learn), pytest |
| 06 | `06_ann_keras` | ANN classification, regression and K-fold cross validation | Keras, scikit-learn |
| 07 | `07_cnn_mnist_keras` | CNN for handwritten digits + upload-and-predict app | Keras, Streamlit |
| 08 | `08_cnn_cifar_pytorch` | CNN for CIFAR-10 colour photos with a hand-written training loop | PyTorch |
| 09 | `09_sentiment_rnn_lstm_gru` | SimpleRNN vs LSTM vs GRU on IMDB movie reviews | Keras |
| 10 | `10_attention_from_scratch` | Self-attention, causal mask, multi-head and positional encoding | NumPy |
| 11 | `11_tiny_gpt_pytorch` | Character-level GPT that writes a short story | PyTorch |

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

Projects 02-04 need [Ollama](https://ollama.com) running with these models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

## How to run

| Project | Command (from inside the folder) |
|---|---|
| 01 | `streamlit run marks_app.py` |
| 02 | `streamlit run chatbot.py` |
| 03 | `python cpp_rag.py` |
| 04 | `streamlit run pdf_qa.py` |
| 05 | `python helpdesk.py`, `streamlit run app.py`, `pytest test_helpdesk.py -v` |
| 06 | `python ann_classification.py`, `python ann_regression.py`, `python kfold_cv.py` |
| 07 | `python train_cnn.py`, then `streamlit run predict_app.py` |
| 08 | `python train.py`, then `python predict.py my_photo.jpg` |
| 09 | `python compare_rnns.py` |
| 10 | `python attention.py` |
| 11 | `python tiny_gpt.py` |

Datasets (MNIST, CIFAR-10, IMDB) download automatically the first time. Trained models are not stored in the repo; run the training script to create them.

## Results (CPU, default settings)

| Project | Result |
|---|---|
| 05 College helpdesk | 10/10 tests pass |
| 06 ANN classification | 95.6% test accuracy |
| 06 ANN regression | R² = 0.41, MAE = 41.8 |
| 06 K-fold (5 folds) | 97.5% mean accuracy |
| 07 MNIST CNN | 98.7% test accuracy (3 epochs) |
| 09 SimpleRNN / LSTM / GRU | 77.0% / 85.7% / 86.1% test accuracy (2 epochs) |
| 11 Tiny GPT | loss 3.42 → 0.13 in 1500 steps |
