# 🔭 DeepLens — ML Document Intelligence
![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20LLaMA3-orange.svg)

---

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Features](#-features)
- [How to Install and Run](#-how-to-install-and-run)
- [How to Use](#-how-to-use)
- [Project Structure](#-project-structure)
- [RAGAS Evaluation](#-ragas-evaluation)
- [Advanced Settings](#-advanced-settings)
- [Where Data Is Saved](#-where-data-is-saved)
- [Tests](#-tests)
- [Contributing](#-contributing)

---

## 📖 Project Description

**What it does:**
DeepLens is a RAG (Retrieval-Augmented Generation) chatbot that lets you upload Machine Learning documents — PDFs, Word docs, slides, CSVs — and ask questions about them in plain English. The AI answers using only your uploaded content, so you always get grounded, traceable answers.

**Why these technologies:**
- **Streamlit** — simple, fast way to build interactive Python apps with no frontend code
- **Groq + LLaMA 3** — extremely fast LLM inference for free, no GPU needed
- **FAISS + BM25** — combining semantic search and keyword search gives far better retrieval than either alone
- **SQLite** — lightweight, zero-config database that keeps chat history across sessions without any setup

**Challenges faced:**
The hardest part was preventing the LLM from making up answers when the context didn't contain enough information. The solution was a strict prompt that forces the model to say "This topic isn't covered in the uploaded documents" rather than hallucinate. RAGAS evaluation was added to automatically detect low-quality answers.

**What makes it stand out:**
Most RAG demos are single-session only. DeepLens persists your uploaded documents and chat history across sessions using SQLite and pickle, so you can close the browser and come back without re-uploading anything.

---

## ✨ Features

| Feature | What it does |
|---|---|
| 📄 Multi-format upload | PDF, DOCX, CSV, PPTX, TXT |
| 🔍 Hybrid retrieval | FAISS (semantic) + BM25 (keyword) combined |
| 🤖 Groq LLM | llama-3.1-8b-instant — fast and free |
| ✏️ Query rewriting | Expands abbreviations, rewrites vague questions |
| ⚡ Smart caching | Skips the LLM for repeated questions (LRU cache) |
| 📊 RAGAS evaluation | Scores every answer: Faithfulness, Relevancy, Precision, Recall |
| 🔢 Token tracking | Shows prompt / completion / total tokens per query |
| 💾 Persistence | Chat history + retriever saved to SQLite — survives page refresh |
| 🕓 Chat history | Sidebar with Starred + Recents, grouped by date like ChatGPT |
| 📥 Export | Download full chat as .txt or .pdf |
| 🧪 Ground truth eval | 10-question preset dataset to benchmark retrieval quality |
| ⚖️ Compare mode | Ask one question across two documents side by side |

---

## 🚀 How to Install and Run

### Prerequisites

- Python 3.10 or higher
- A free Groq API key from https://console.groq.com

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add your Groq API key

Create a `.env` file in the project root folder:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Step 5 — Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🖥️ How to Use

1. **Upload documents** — click "Choose files" in the left sidebar and select your ML PDFs or docs
2. **Process** — click the "🚀 Process Documents" button and wait for the progress bar to complete
3. **Chat** — type any question about your documents in the chat box at the bottom
4. **View sources** — expand "📎 Source Citations" under any answer to see exactly which page it came from
5. **Check quality** — the RAGAS panel under each answer shows Faithfulness, Relevancy, Precision, and Recall scores
6. **Re-ask from history** — click any past question in the sidebar to ask it again instantly
7. **Export** — scroll to the bottom of the sidebar to download your full chat as .txt or .pdf

### Example questions you can ask

```
What is backpropagation?
Explain the difference between CNN and RNN
What are the main types of activation functions?
Summarise what chapter 3 covers
Compare what both documents say about gradient descent
```

![image alt](https://github.com/Siridasari54/rag-chatbot/blob/5c9a21233e65bca8068a1d31e816c9a70ecdc27c/rag%20image.png)

### Credentials

No login required. Your Groq API key in the `.env` file is the only credential needed.

---

## 🗂️ Project Structure

```
rag/
├── app.py                        # Entry point — starts the Streamlit app
├── requirements.txt              # All Python dependencies
├── .env                          # Your GROQ_API_KEY (never commit this)
├── .gitignore
│
├── config/
│   └── settings.py               # Session state defaults and init function
│
├── loaders/
│   └── file_loader.py            # Handles PDF / DOCX / CSV / PPTX / TXT
│
├── rag/
│   ├── chains.py                 # SimpleConvChain — custom LLM chain with history
│   ├── embeddings.py             # HuggingFace embeddings + FAISS index builder
│   ├── retriever.py              # Hybrid BM25 + FAISS retriever
│   ├── query_rewriter.py         # Abbreviation expansion + LLM-based rewriting
│   └── summarizer.py             # One-line document summarizer
│
├── evaluation/
│   ├── ragas_eval.py             # LLM-as-judge RAGAS scoring (4 metrics)
│   └── eval_dataset.py           # 10 preset ground-truth QA pairs
│
├── services/
│   └── groq_services.py          # Builds the ChatGroq LLM instance
│
├── ui/
│   ├── sidebar.py                # Upload, settings, history, export
│   ├── chat_ui.py                # Chat rendering, query handling, follow-ups
│   └── styles.py                 # Dark theme CSS
│
└── utils/
    ├── cache.py                  # LRU query cache with TTL and eviction
    ├── confidence.py             # Relevance score heuristic + bar HTML
    ├── helpers.py                # Sentence window expand, source renderer
    ├── persistence.py            # SQLite + pickle read/write functions
    ├── text_cleaner.py           # Strips source tags from LLM output
    └── token_counter.py          # tiktoken-based token counter
```

---

## 📊 RAGAS Evaluation

Every answer is automatically scored by a separate judge LLM on 4 metrics:

| Metric | What it checks |
|---|---|
| **Faithfulness** | Is the answer actually supported by the retrieved context? |
| **Answer Relevancy** | Does the answer address what was asked? |
| **Context Precision** | How much of the retrieved context was actually useful? |
| **Context Recall** | Did the answer cover all the key points it should? |

Scores range from 0.0 (bad) to 1.0 (perfect).
Color coding: 🟢 ≥ 70% &nbsp;·&nbsp; 🟡 40–70% &nbsp;·&nbsp; 🔴 below 40%

---

## 🔧 Advanced Settings

Open "⚙️ Advanced Settings" in the sidebar to tune these:

| Setting | Default | What it controls |
|---|---|---|
| Chunk Size | 300 | How many characters fit in one text chunk |
| Chunk Overlap | 100 | How much two adjacent chunks share for continuity |
| Top-K Retrieval | 5 | How many chunks are fetched per query |
| Query Cache | ON | Returns cached answers for repeated questions |
| Query Rewriting | ON | Rewrites short or vague queries before searching |
| RAGAS Evaluation | ON | Scores every answer (adds ~2–5 s per query) |

---

## 📁 Where Data Is Saved

| What | Location |
|---|---|
| Uploaded files | `data/` folder |
| FAISS vector index | `data/faiss_index.pkl` |
| Text chunks | `data/chunks.pkl` |
| Chat history | `data/deeplens.db` (SQLite) |
| Document metadata | `data/deeplens.db` (SQLite) |

Everything is restored automatically when you reopen the app — no need to re-upload documents.

> **Note:** Add `data/` to your `.gitignore` so you don't accidentally push large files or your personal chat history to GitHub.

---

## 🧪 Tests

A built-in ground truth evaluation is included. It runs 10 preset ML questions through the retriever and scores results automatically.

**To run it:**
1. Upload at least one ML document and click Process Documents
2. In the sidebar scroll to "🧪 Ground Truth Evaluation"
3. Click "▶️ Run Eval Dataset"

It reports how many of the 10 questions were successfully retrieved and the average word-overlap score against known correct answers.

**To run the retriever test script manually:**

```bash
python test_retriever.py
```

This loads `machine_learning.pdf`, chunks it, builds the retriever, and prints the top 5 retrieved chunks for the query "what is machine learning".

---

## 🤝 Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature/your-feature-name
```

3. Make your changes and commit

```bash
git commit -m "add: your feature description"
```

4. Push to your branch

```bash
git push origin feature/your-feature-name
```

5. Open a Pull Request on GitHub

Please make sure your code runs without errors before submitting. If you are adding a new feature, update this README with any relevant usage instructions.

---

Built with ❤️ using Streamlit · LangChain · Groq · FAISS · SQLite
