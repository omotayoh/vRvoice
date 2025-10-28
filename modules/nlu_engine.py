from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import torch

# -----------------------------
# Load Intent Classifier
# -----------------------------
device = 0 if torch.cuda.is_available() else -1
intent_classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased",
    device=device
)

# -----------------------------
# Load Sentence Transformer
# -----------------------------
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Load Command Map
# -----------------------------
with open("commands/command_map.json", "r") as f:
    COMMAND_MAP = json.load(f)

phrases = list(COMMAND_MAP.keys())

# Precompute embeddings for FAISS
phrase_embeddings = semantic_model.encode(phrases, convert_to_numpy=True)
dimension = phrase_embeddings.shape[1]

# Build FAISS index
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
# Normalize embeddings for cosine similarity
faiss.normalize_L2(phrase_embeddings)
index.add(phrase_embeddings)

# -----------------------------
# Functions
# -----------------------------
def classify_intent(text):
    result = intent_classifier(text)[0]
    return result['label'], result['score']

def semantic_match(text, top_k=1):
    text_embedding = semantic_model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(text_embedding)
    scores, indices = index.search(text_embedding, top_k)
    best_score = float(scores[0][0])
    best_phrase = phrases[indices[0][0]]
    if best_score >= 0.65:
        return best_phrase, best_score
    return None, None

def get_command(text):
    # First try intent classification
    label, score = classify_intent(text)
    if score >= 0.75 and label in COMMAND_MAP:
        return COMMAND_MAP[label], label, score

    # Fallback to FAISS semantic similarity
    match, sim_score = semantic_match(text)
    if match:
        return COMMAND_MAP[match], match, sim_score

    return None, None, None