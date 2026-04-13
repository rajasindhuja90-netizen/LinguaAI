"""
LinguaAI — NLP Chatbot (Python Version)
=========================================
A terminal-based NLP assistant powered by Anthropic Claude API.
Mirrors the full functionality of the HTML/JS version:
  - Built-in NLP Knowledge Base (instant offline answers)
  - Claude API fallback for unanswered queries
  - Smart fallback when API is unavailable
  - Multi-turn session memory
  - 14 NLP topic suggestions

Requirements:
    pip install anthropic

Usage:
    python linguaai_chatbot.py
    python linguaai_chatbot.py --api-key sk-ant-...
"""

import os
import re
import sys
import math
import json
import argparse
from collections import Counter

# ── Optional: Anthropic SDK ──────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── ANSI Colors ───────────────────────────────────────────────────
class C:
    TEAL    = "\033[36m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
    RED     = "\033[91m"

# ─────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE  (matches the HTML KB object exactly)
# ─────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = {

    "what is nlp": """
WHAT IS NLP (Natural Language Processing)?
===========================================
Natural Language Processing (NLP) is a branch of AI that enables computers
to understand, interpret, generate, and interact with human language.

Key Goals:
  • Enable machines to read and understand text
  • Allow computers to generate human-like language
  • Bridge human communication and machine understanding

Core NLP Tasks:
  Task                Description                      Example
  ──────────────────────────────────────────────────────────────
  Tokenisation        Split text into words/sentences  "I love NLP" → ["I","love","NLP"]
  POS Tagging         Identify grammatical roles       "love" → VERB
  NER                 Find named entities              "Delhi" → LOCATION
  Sentiment Analysis  Detect emotion/opinion           "Great!" → POSITIVE
  Machine Translation Translate between languages      English → Hindi
  Text Summarisation  Shorten documents                Article → 3 key points
  Question Answering  Answer natural questions         "What is NLP?" → answer

Real-World Applications:
  🤖 Chatbots & Virtual Assistants (Siri, Alexa, ChatGPT)
  📧 Spam Detection
  🔍 Search Engines (Google uses NLP)
  🌍 Google Translate
  📰 News Summarisation
  💊 Medical Record Analysis

NLP Pipeline:
  Input Text → Tokenisation → POS Tagging → NER → Parsing → Understanding → Response
""",

    "bert vs gpt": """
BERT vs GPT — Key Differences
==============================
Feature          BERT                              GPT
────────────────────────────────────────────────────────────────
Full Name        Bidirectional Encoder             Generative Pre-trained
                 Representations from Transformers Transformer
Developer        Google (2018)                     OpenAI (2018–present)
Architecture     Encoder-only                      Decoder-only
Direction        Bidirectional (left AND right)    Unidirectional (left → right)
Pre-training     Masked Language Model (MLM)+NSP   Causal Language Modeling
Best For         Understanding tasks               Generation tasks
Output           Contextual embeddings             Generated next tokens

BERT Example:
─────────────
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "NLP is amazing"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # [1, seq_len, 768]

GPT Example:
────────────
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_ids = tokenizer.encode("NLP is", return_tensors='pt')
output = model.generate(input_ids, max_length=20)
print(tokenizer.decode(output[0]))

When to use which?
  • BERT  → Sentiment analysis, NER, question answering, classification
  • GPT   → Text generation, chatbots, code generation, creative writing
""",

    "attention mechanism": """
Attention Mechanism in NLP
===========================
Attention lets models focus on RELEVANT parts of input when producing output.
It solves the bottleneck problem of older RNN encoder-decoder models.

Formula:
  Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

  Q = Query (what we're looking for)
  K = Key   (what each position offers)
  V = Value (actual content to retrieve)

Python Implementation:
──────────────────────
import torch
import torch.nn.functional as F

def self_attention(Q, K, V):
    d_k = Q.shape[-1]

    # Step 1: Score (dot product of Query and Key)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # Step 2: Softmax to get weights (probabilities)
    weights = F.softmax(scores, dim=-1)

    # Step 3: Weighted sum of Values
    output = torch.matmul(weights, V)
    return output, weights

# Example
seq_len, d_model = 5, 64
Q = torch.randn(1, seq_len, d_model)
K = torch.randn(1, seq_len, d_model)
V = torch.randn(1, seq_len, d_model)

out, attn_weights = self_attention(Q, K, V)
print("Output shape:", out.shape)         # [1, 5, 64]
print("Attention weights:", attn_weights.shape)  # [1, 5, 5]

Types of Attention:
  Type             Used In         Description
  ───────────────────────────────────────────────────────────────
  Self-Attention   Transformers    Each token attends to all tokens
  Cross-Attention  Enc-Dec models  Decoder attends to encoder output
  Causal Attention GPT             Each token only attends to past tokens
  Multi-Head       BERT, GPT       Multiple parallel attention heads
""",

    "tfidf": """
TF-IDF (Term Frequency – Inverse Document Frequency)
======================================================
TF-IDF reflects how important a word is to a document in a corpus.

Formula:
  TF(t, d)  = count(t in d) / total_terms(d)
  IDF(t, D) = log(N / docs_containing_t)
  TF-IDF    = TF × IDF

Python — From Scratch:
──────────────────────
import math
from collections import Counter

def tf(word, doc):
    return doc.count(word) / len(doc)

def idf(word, docs):
    n = sum(1 for doc in docs if word in doc)
    return math.log(len(docs) / (1 + n))

def tfidf(word, doc, docs):
    return tf(word, doc) * idf(word, docs)

docs = [
    "NLP is amazing and NLP is fun",
    "Deep learning powers NLP models",
    "BERT is a transformer model for NLP"
]
tokenised = [doc.lower().split() for doc in docs]

print("TF-IDF scores for Document 1:")
for word in set(tokenised[0]):
    score = tfidf(word, tokenised[0], tokenised)
    print(f"  {word:15s}: {score:.4f}")

Python — Using sklearn:
───────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)
print(df.round(3))

Key Insight:
  • High TF-IDF → frequent in THIS doc, rare across corpus → important
  • Low TF-IDF  → common everywhere ("the", "is") → not useful
""",

    "ner spacy": """
Named Entity Recognition (NER) with spaCy
==========================================
NER identifies and classifies named entities: people, orgs, locations, dates...

spaCy Entity Labels:
  PERSON   → People (real or fictional)
  ORG      → Organisations, companies
  GPE      → Countries, cities, states
  DATE     → Dates and time periods
  MONEY    → Monetary values
  PRODUCT  → Products and items
  EVENT    → Named events

Basic NER Example:
──────────────────
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Elon Musk founded SpaceX in 2002 in California with $100 million."

doc = nlp(text)
print("Entities found:")
for ent in doc.ents:
    print(f"  {ent.text:20s} → {ent.label_:10s} ({spacy.explain(ent.label_)})")

# Output:
#   Elon Musk            → PERSON     (People, including fictional)
#   SpaceX               → ORG        (Companies, agencies, institutions)
#   2002                 → DATE       (Absolute or relative dates)
#   California           → GPE        (Countries, cities, states)
#   $100 million         → MONEY      (Monetary values)

Custom NER Training:
────────────────────
import spacy
from spacy.training import Example

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")
ner.add_label("NLP_TERM")

TRAIN_DATA = [
    ("BERT is a transformer model", {"entities": [(0, 4, "NLP_TERM")]}),
    ("Word2Vec creates word embeddings", {"entities": [(0, 8, "NLP_TERM")]}),
]

nlp.begin_training()
for epoch in range(20):
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example])

doc = nlp("BERT is used for NLP tasks")
for ent in doc.ents:
    print(ent.text, "→", ent.label_)
""",

    "sentiment analysis": """
Sentiment Analysis
==================
Detects emotional tone: POSITIVE, NEGATIVE, or NEUTRAL.

Method 1 — VADER (Best for social media):
──────────────────────────────────────────
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

texts = [
    "NLP is absolutely amazing! I love it 😍",
    "This model is terrible and very slow.",
    "The accuracy is okay, nothing special.",
]

for text in texts:
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    sentiment = ('POSITIVE' if compound >= 0.05
                 else 'NEGATIVE' if compound <= -0.05
                 else 'NEUTRAL')
    print(f"Text: {text[:40]}")
    print(f"  Scores: {scores}")
    print(f"  Sentiment: {sentiment}\\n")

Method 2 — TextBlob (Simple & fast):
──────────────────────────────────────
from textblob import TextBlob

texts = ["NLP is amazing!", "I hate bugs.", "Python is okay."]

for text in texts:
    blob = TextBlob(text)
    p = blob.sentiment.polarity
    label = "POSITIVE" if p > 0 else "NEGATIVE" if p < 0 else "NEUTRAL"
    print(f"{text:35s} → {label} (polarity={p:.2f})")

Method 3 — BERT-based (Most accurate):
────────────────────────────────────────
from transformers import pipeline

classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english")

results = classifier(["NLP is amazing!", "This model is terrible."])
for r in results:
    print(f"Label: {r['label']:10s}  Score: {r['score']:.4f}")
""",

    "tokenisation": """
Tokenisation in NLP
====================
Splitting raw text into meaningful units (tokens): words, subwords, or characters.

1. Word Tokenisation (NLTK):
─────────────────────────────
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

text = "NLP is amazing! It powers chatbots, translation, and more."
words = word_tokenize(text)
sents = sent_tokenize(text)

print("Words:", words)
print("Sentences:", sents)

2. spaCy Tokenisation:
───────────────────────
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("I'm learning NLP with Python!")
for token in doc:
    print(f"  {token.text:15s} lemma={token.lemma_:15s} pos={token.pos_}")

3. Subword Tokenisation (BERT — WordPiece):
────────────────────────────────────────────
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("unbelievable tokenisation")
ids    = tokenizer.encode("unbelievable tokenisation")

print("Tokens:", tokens)
# ['un', '##believ', '##able', 'token', '##isation']
print("IDs:", ids)

4. Stemming vs Lemmatisation:
──────────────────────────────
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "flies", "better", "studies"]
for word in words:
    print(f"  {word:12s} stem={stemmer.stem(word):12s} "
          f"lemma={lemmatizer.lemmatize(word)}")
""",

    "word2vec": """
Word2Vec Embeddings
====================
Word2Vec maps words to dense numerical vectors preserving semantic meaning.
Similar words → similar vectors (nearby in vector space).

Two Architectures:
  • CBOW (Continuous Bag of Words) — predicts target from context words
  • Skip-gram                     — predicts context words from target

Training with Gensim:
─────────────────────
from gensim.models import Word2Vec

# Sample corpus
corpus = [
    ["king", "rules", "the", "kingdom"],
    ["queen", "rules", "the", "kingdom"],
    ["man", "is", "strong"],
    ["woman", "is", "strong"],
    ["paris", "is", "capital", "of", "france"],
    ["berlin", "is", "capital", "of", "germany"],
]

model = Word2Vec(
    sentences=corpus,
    vector_size=100,  # embedding dimensions
    window=5,         # context window size
    min_count=1,
    workers=4,
    sg=1              # 1=Skip-gram, 0=CBOW
)
model.save("word2vec.model")

# Find similar words
print(model.wv.most_similar("king", topn=3))

# Word arithmetic: king - man + woman ≈ queen
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)
print("king - man + woman =", result)

# Vector for a word
vec = model.wv["nlp"]
print("Vector shape:", vec.shape)  # (100,)
print("Similarity:", model.wv.similarity("king", "queen"))
""",

    "transformer": """
Transformer Architecture
=========================
"Attention Is All You Need" (Vaswani et al., 2017) — no RNNs, only attention.

Key Components:
  1. Input Embeddings + Positional Encoding
  2. Multi-Head Self-Attention
  3. Feed-Forward Network
  4. Layer Normalization + Residual Connections

Complete Transformer Block (PyTorch):
──────────────────────────────────────
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-Head Attention + residual
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.drop(attn_out))
        # Feed-Forward + residual
        x = self.norm2(x + self.drop(self.ff(x)))
        return x

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Test
block = TransformerBlock(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
out = block(x)
print("Output shape:", out.shape)  # [2, 10, 512]
""",

    "lstm": """
LSTM vs GRU
============
Both solve the vanishing gradient problem of vanilla RNNs.

LSTM (Long Short-Term Memory) — 3 gates:
──────────────────────────────────────────
  Forget Gate : decides what to forget from cell state
  Input Gate  : decides what new info to add
  Output Gate : decides what to output

import torch
import torch.nn as nn

# LSTM
lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2,
               batch_first=True, dropout=0.2)

x = torch.randn(32, 20, 10)  # (batch, seq_len, features)
output, (h_n, c_n) = lstm(x)
print("LSTM output:", output.shape)  # [32, 20, 64]
print("Hidden state:", h_n.shape)    # [2, 32, 64]

GRU (Gated Recurrent Unit) — 2 gates (lighter than LSTM):
───────────────────────────────────────────────────────────
  Reset Gate  : controls how much past info to forget
  Update Gate : controls how much past state to keep

gru = nn.GRU(input_size=10, hidden_size=64, num_layers=2,
             batch_first=True, dropout=0.2)

output, h_n = gru(x)
print("GRU output:", output.shape)  # [32, 20, 64]

Comparison:
  Feature          LSTM           GRU
  ─────────────────────────────────────
  Gates            3              2
  Parameters       More           Fewer
  Speed            Slower         Faster
  Memory           Cell + Hidden  Hidden only
  Performance      Slightly better on long seqs  Similar on short
  Use Case         Long sequences, NLP  Speed-critical tasks

Text Classification with LSTM:
────────────────────────────────
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embed(x)
        _, (h_n, _) = self.lstm(emb)
        return self.fc(h_n[-1])
""",

    "text classification": """
Text Classification
====================
Assigning predefined categories to text documents.

1. Naive Bayes (Simple baseline):
──────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

texts = [
    "I love this product", "Amazing experience",
    "Terrible quality", "Worst purchase ever",
    "Okay, nothing special", "Average performance",
]
labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf',   MultinomialNB()),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Predict new text
print(pipeline.predict(["This is wonderful!"]))  # ['positive']

2. BERT Fine-tuning (State of the art):
─────────────────────────────────────────
from transformers import (BertTokenizer, BertForSequenceClassification,
                          Trainer, TrainingArguments)
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model     = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=3)

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True,
                     max_length=128, return_tensors='pt')

inputs = tokenize(["NLP is great!", "This is terrible"])
outputs = model(**inputs)
probs   = torch.softmax(outputs.logits, dim=1)
print("Class probabilities:", probs.detach().numpy())
""",

    "pos tagging": """
POS Tagging (Part-of-Speech Tagging)
======================================
Labels each word with its grammatical role: NOUN, VERB, ADJ, ADV, etc.

Common POS Tags:
  NN   → Noun (singular)      NNS  → Noun (plural)
  VB   → Verb (base form)     VBG  → Verb (gerund)
  JJ   → Adjective            RB   → Adverb
  DT   → Determiner           IN   → Preposition
  PRP  → Personal Pronoun     CC   → Coordinating Conjunction

1. NLTK POS Tagging:
─────────────────────
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

text = "The quick brown fox jumps over the lazy dog"
tokens = nltk.word_tokenize(text)
tags   = nltk.pos_tag(tokens)

for word, tag in tags:
    print(f"  {word:12s} → {tag}")

2. spaCy POS Tagging (Recommended):
─────────────────────────────────────
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("LinguaAI quickly processes complex NLP text.")
for token in doc:
    print(f"  {token.text:15s} POS={token.pos_:8s} "
          f"Tag={token.tag_:8s} Dep={token.dep_}")

3. Custom Feature Extraction using POS:
─────────────────────────────────────────
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_nouns(text):
    doc = nlp(text)
    return [t.text for t in doc if t.pos_ == "NOUN"]

def extract_verbs(text):
    doc = nlp(text)
    return [t.lemma_ for t in doc if t.pos_ == "VERB"]

text = "The scientist quickly discovers a new NLP algorithm"
print("Nouns:", extract_nouns(text))
print("Verbs:", extract_verbs(text))
""",

    "language models": """
Language Models in NLP
=======================
A Language Model (LM) assigns probabilities to sequences of words.
P("I love NLP") > P("NLP love I")

Types:
  N-gram LM      → Statistical, counts word co-occurrences
  RNN/LSTM LM    → Neural, sequential processing
  Transformer LM → Attention-based (BERT, GPT, T5)

N-gram Language Model:
──────────────────────
from collections import defaultdict, Counter
import random

class NgramLM:
    def __init__(self, n=2):
        self.n = n
        self.model = defaultdict(Counter)

    def train(self, corpus):
        for sentence in corpus:
            tokens = sentence.lower().split()
            for i in range(len(tokens) - self.n + 1):
                ctx    = tuple(tokens[i:i+self.n-1])
                target = tokens[i+self.n-1]
                self.model[ctx][target] += 1

    def predict_next(self, context):
        ctx = tuple(context[-self.n+1:])
        if ctx not in self.model:
            return "<UNK>"
        return self.model[ctx].most_common(1)[0][0]

    def generate(self, seed, length=10):
        tokens = seed.lower().split()
        for _ in range(length):
            tokens.append(self.predict_next(tokens))
        return " ".join(tokens)

corpus = [
    "NLP is amazing and NLP is useful",
    "deep learning powers NLP models",
    "transformer models changed NLP forever",
    "BERT is a powerful NLP model",
]

lm = NgramLM(n=2)
lm.train(corpus)
print(lm.generate("NLP is", length=5))

GPT-2 Text Generation:
───────────────────────
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model     = GPT2LMHeadModel.from_pretrained('gpt2')

prompt    = "Natural language processing is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=60,
    num_beams=5,
    no_repeat_ngram_size=2,
    temperature=0.8,
    do_sample=True,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
""",

    "text summarisation": """
Text Summarisation
==================
Automatically generate a concise summary of a long document.

Two main approaches:
  • Extractive  — picks key sentences from original text
  • Abstractive — generates new sentences (like a human)

1. Extractive — TF-IDF Sentence Scoring:
─────────────────────────────────────────
import math
from collections import Counter
import re

def summarise_extractive(text, top_n=3):
    # Tokenise
    sentences = re.split(r'(?<=[.!?]) +', text)
    words = re.findall(r'\\w+', text.lower())
    stop_words = {'the','is','a','an','and','or','in','of','to','it','that','this'}

    # TF scores (exclude stop words)
    freq = Counter(w for w in words if w not in stop_words)
    total = sum(freq.values())
    tf = {w: c/total for w, c in freq.items()}

    # Score sentences
    def score(sent):
        ws = re.findall(r'\\w+', sent.lower())
        return sum(tf.get(w, 0) for w in ws if w not in stop_words)

    ranked = sorted(sentences, key=score, reverse=True)
    return ' '.join(ranked[:top_n])

text = '''
Natural Language Processing (NLP) is a fascinating field of AI.
It combines linguistics and machine learning to understand human language.
NLP powers applications like chatbots, search engines, and translators.
Modern NLP uses deep learning with transformer models like BERT and GPT.
These models achieve state-of-the-art results on many language tasks.
'''
print(summarise_extractive(text, top_n=2))

2. Abstractive — HuggingFace BART:
────────────────────────────────────
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

long_text = (
    "Natural Language Processing (NLP) enables computers to understand "
    "human language. It is used in chatbots, search engines, translators, "
    "and many other applications. Modern NLP uses transformer architectures "
    "like BERT and GPT to achieve state-of-the-art results on language tasks."
)

summary = summarizer(long_text, max_length=50, min_length=10, do_sample=False)
print("Summary:", summary[0]['summary_text'])
""",
}

# ─────────────────────────────────────────────────────────────────
# KEYWORD → KB ROUTING  (mirrors matchKB + generalNLP in JS)
# ─────────────────────────────────────────────────────────────────
KB_ROUTING = {
    "what is nlp":        ["what is nlp", "define nlp", "nlp meaning", "nlp stands for",
                           "natural language processing"],
    "bert vs gpt":        ["bert", "gpt", "bert vs", "vs gpt", "compare bert"],
    "attention mechanism":["attention", "self-attention", "multi-head", "attention mechanism"],
    "tfidf":              ["tfidf", "tf-idf", "tf idf", "term frequency"],
    "ner spacy":          ["ner", "named entity", "spacy", "entity recognition"],
    "sentiment analysis": ["sentiment", "opinion mining", "vader", "textblob",
                           "positive negative"],
    "tokenisation":       ["tokeniz", "tokenis", "stemming", "lemmatiz", "nltk tokenize"],
    "word2vec":           ["word2vec", "word embedding", "word vector", "gensim",
                           "skip-gram", "cbow"],
    "transformer":        ["transformer", "attention is all", "encoder decoder",
                           "positional encoding"],
    "lstm":               ["lstm", "gru", "recurrent", "rnn", "vanishing gradient"],
    "text classification":["text classif", "document classif", "naive bayes",
                           "classif", "categoriz"],
    "pos tagging":        ["pos tag", "part of speech", "grammatical", "noun verb",
                           "pos label"],
    "language models":    ["language model", "n-gram", "ngram", "perplexity",
                           "text generation"],
    "text summarisation": ["summar", "abstractive", "extractive", "bart",
                           "key sentences"],
}

NLP_TOPICS = [
    "What is NLP?", "Tokenisation with NLTK", "BERT vs GPT",
    "Attention mechanism", "TF-IDF Python code", "Sentiment analysis",
    "Named Entity Recognition", "Word2Vec embeddings",
    "Transformer architecture", "LSTM vs GRU", "Text classification",
    "POS tagging", "Language models", "Text summarisation",
]


def match_kb(query: str) -> str | None:
    """Return KB answer if query matches any keyword pattern."""
    q = query.lower()
    for kb_key, keywords in KB_ROUTING.items():
        if any(kw in q for kw in keywords):
            return KNOWLEDGE_BASE.get(kb_key)
    return None


# ─────────────────────────────────────────────────────────────────
# CLAUDE API  (mirrors callClaude in JS)
# ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are LinguaAI, an expert NLP (Natural Language Processing) assistant. "
    "Answer any NLP question with detailed, well-structured responses. "
    "Include Python code examples when relevant. Cover topics like tokenisation, "
    "BERT, GPT, transformers, attention, embeddings, sentiment analysis, NER, "
    "text classification, and all NLP concepts. Be educational, thorough, and "
    "technically accurate."
)

def call_claude(history: list[dict], api_key: str) -> str | None:
    """Send conversation to Anthropic API and return reply text."""
    if not ANTHROPIC_AVAILABLE:
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=history,
        )
        return response.content[0].text if response.content else None
    except Exception as e:
        print(f"{C.RED}  [API Error] {e}{C.RESET}")
        return None


def smart_fallback(query: str) -> str:
    """Mirrors generateSmartFallback in JS."""
    topics_str = "\n".join(f"  • {t}" for t in NLP_TOPICS)
    return (
        f'NLP Answer for: "{query}"\n'
        f"{'─'*50}\n"
        "This is a great NLP question! To get a full AI-powered answer,\n"
        "provide your Anthropic API key with --api-key or ANTHROPIC_API_KEY env var.\n\n"
        "Topics I can answer instantly (no API needed):\n"
        f"{topics_str}\n\n"
        "Just type any of those topics!"
    )


# ─────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────
def print_banner():
    print(f"""
{C.TEAL}{C.BOLD}╔══════════════════════════════════════════════════════╗
║   🧠  LinguaAI — NLP Expert Chatbot (Python)         ║
║   AI-Powered Conversational System using NLP         ║
╚══════════════════════════════════════════════════════╝{C.RESET}
{C.GRAY}Type your NLP question or one of the suggestions below.
Commands: /topics · /new · /history · /quit{C.RESET}
""")


def print_suggestions():
    print(f"\n{C.CYAN}{C.BOLD}💡 Suggested Topics:{C.RESET}")
    for i, t in enumerate(NLP_TOPICS, 1):
        print(f"  {C.GRAY}{i:2d}.{C.RESET} {t}")
    print()


def print_user(text: str):
    print(f"\n{C.BLUE}{C.BOLD}You:{C.RESET}")
    print(f"  {text}\n")


def print_bot(text: str):
    print(f"{C.TEAL}{C.BOLD}🧠 LinguaAI:{C.RESET}")
    # Simple formatting: bold **text**, strip markdown headers to plain
    lines = text.strip().split("\n")
    for line in lines:
        line = re.sub(r'\*\*(.+?)\*\*', f'{C.BOLD}\\1{C.RESET}', line)
        line = re.sub(r'^#+\s*', f'{C.TEAL}{C.BOLD}', line)
        if re.match(r'^#{1,3} ', text.split("\n")[0] if text else ""):
            pass
        print(f"  {line}{C.RESET}")
    print()


def print_source(source: str):
    labels = {
        "kb":       f"{C.GREEN}[Knowledge Base — instant]{C.RESET}",
        "claude":   f"{C.MAGENTA}[Claude API]{C.RESET}",
        "fallback": f"{C.YELLOW}[Fallback — add API key for full answers]{C.RESET}",
    }
    print(f"  {labels.get(source, '')}\n")


# ─────────────────────────────────────────────────────────────────
# CHAT SESSION
# ─────────────────────────────────────────────────────────────────
class LinguaAI:
    def __init__(self, api_key: str = ""):
        self.api_key  = api_key
        self.history: list[dict] = []
        self.sessions: list[dict] = []

    def chat(self, user_input: str) -> tuple[str, str]:
        """
        Process a user message. Returns (reply, source).
        source: 'kb' | 'claude' | 'fallback'
        """
        self.history.append({"role": "user", "content": user_input})

        # 1. Try knowledge base first (instant, offline)
        reply = match_kb(user_input)
        source = "kb"

        # 2. Try Claude API
        if not reply and self.api_key:
            reply = call_claude(self.history, self.api_key)
            source = "claude"

        # 3. Smart fallback
        if not reply:
            reply = smart_fallback(user_input)
            source = "fallback"

        self.history.append({"role": "assistant", "content": reply})
        return reply, source

    def reset(self):
        self.history = []


# ─────────────────────────────────────────────────────────────────
# MAIN CLI LOOP
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LinguaAI — NLP Expert Chatbot (Python)"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    args = parser.parse_args()

    bot = LinguaAI(api_key=args.api_key)

    print_banner()

    if bot.api_key:
        print(f"{C.GREEN}✅ API key loaded — full AI responses active!{C.RESET}\n")
    else:
        print(f"{C.YELLOW}⚠  No API key set. Knowledge Base answers only.{C.RESET}")
        print(f"{C.GRAY}   Use: --api-key sk-ant-... or export ANTHROPIC_API_KEY=...{C.RESET}\n")

    print_suggestions()

    while True:
        try:
            user_input = input(f"{C.BOLD}You > {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.GRAY}Goodbye! 👋{C.RESET}")
            break

        if not user_input:
            continue

        # Commands
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "quit", "exit", "q"):
            print(f"{C.GRAY}Goodbye! 👋{C.RESET}")
            break
        elif cmd in ("/topics", "topics"):
            print_suggestions()
            continue
        elif cmd in ("/new", "new chat"):
            bot.reset()
            print(f"{C.GREEN}✓ New chat started.{C.RESET}\n")
            continue
        elif cmd in ("/history", "history"):
            if not bot.history:
                print(f"{C.GRAY}  No history yet.{C.RESET}\n")
            else:
                print(f"\n{C.BOLD}Chat History ({len(bot.history)//2} turns):{C.RESET}")
                for msg in bot.history:
                    role = "You" if msg["role"] == "user" else "LinguaAI"
                    preview = msg["content"][:80].replace("\n", " ")
                    print(f"  {C.BOLD}{role}:{C.RESET} {preview}{'...' if len(msg['content'])>80 else ''}")
                print()
            continue

        # Handle numeric topic selection
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(NLP_TOPICS):
                user_input = NLP_TOPICS[idx]
                print(f"{C.GRAY}  → Selected: {user_input}{C.RESET}")
            else:
                print(f"{C.YELLOW}  Please enter 1–{len(NLP_TOPICS)}{C.RESET}\n")
                continue

        print_user(user_input)
        reply, source = bot.chat(user_input)
        print_bot(reply)
        print_source(source)


if __name__ == "__main__":
    main()
