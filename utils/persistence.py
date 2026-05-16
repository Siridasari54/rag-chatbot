# utils/persistence.py — NEW FILE

import os
import json
import sqlite3
import pickle
import streamlit as st

DB_PATH    = "data/deeplens.db"
FAISS_PATH = "data/faiss_index.pkl"
CHUNKS_PATH = "data/chunks.pkl"


# ── Database setup ─────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question    TEXT,
            rewritten   TEXT,
            answer      TEXT,
            confidence  REAL,
            from_cache  INTEGER,
            token_usage TEXT,
            ragas_scores TEXT,
            timestamp   REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_docs (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT UNIQUE,
            pages    INTEGER,
            summary  TEXT
        )
    """)

    conn.commit()
    conn.close()


# ── Chat history ───────────────────────────────────────────────────────────────

def save_chat(chat: dict):
    """Save a single chat entry to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        INSERT INTO chat_history
        (question, rewritten, answer, confidence, from_cache, token_usage, ragas_scores, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        chat.get("question", ""),
        chat.get("rewritten_query", ""),
        chat.get("answer", ""),
        chat.get("confidence", 0),
        int(chat.get("from_cache", False)),
        json.dumps(chat.get("token_usage", {})),
        json.dumps(chat.get("ragas_scores") or {}),
        chat.get("timestamp", 0),
    ))
    conn.commit()
    conn.close()


def load_chat_history() -> list:
    """Load all chat history from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT * FROM chat_history ORDER BY timestamp ASC")
    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "question":        row[1],
            "rewritten_query": row[2],
            "answer":          row[3],
            "confidence":      row[4],
            "from_cache":      bool(row[5]),
            "token_usage":     json.loads(row[6]) if row[6] else {},
            "ragas_scores":    json.loads(row[7]) if row[7] else None,
            "timestamp":       row[8],
            "sources":         [],   # sources not persisted (too large)
        })
    return history


def clear_chat_history():
    """Delete all chat history from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()


# ── Uploaded docs ──────────────────────────────────────────────────────────────

def save_doc_meta(name: str, pages: int, summary: str):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO uploaded_docs (name, pages, summary)
        VALUES (?, ?, ?)
    """, (name, pages, summary))
    conn.commit()
    conn.close()


def load_doc_meta() -> list:
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT name, pages, summary FROM uploaded_docs")
    rows = c.fetchall()
    conn.close()
    return [{"name": r[0], "pages": r[1], "summary": r[2]} for r in rows]


def clear_doc_meta():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("DELETE FROM uploaded_docs")
    conn.commit()
    conn.close()


# ── FAISS index + chunks ───────────────────────────────────────────────────────

def save_retriever(retriever, chunks: list):
    """Pickle the retriever and chunks to disk."""
    with open(FAISS_PATH,  "wb") as f:
        pickle.dump(retriever, f)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_retriever():
    """Load retriever and chunks from disk. Returns (retriever, chunks) or (None, [])."""
    if os.path.exists(FAISS_PATH) and os.path.exists(CHUNKS_PATH):
        with open(FAISS_PATH,  "rb") as f:
            retriever = pickle.load(f)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return retriever, chunks
    return None, []


def retriever_exists() -> bool:
    return os.path.exists(FAISS_PATH) and os.path.exists(CHUNKS_PATH)