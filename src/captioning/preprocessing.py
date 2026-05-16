"""
Caption preprocessing for Flickr8k.
"""

import re
import json
import numpy as np
from pathlib import Path
from collections import Counter

CAPTIONS_FILE   = Path("data/flickr8k/captions.txt")
VOCAB_FILE      = Path("data/flickr8k/vocab.json")
SPLIT_DIR       = Path("data/flickr8k")

SPECIAL_TOKENS  = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
SPECIAL_COUNT   = len(SPECIAL_TOKENS)

N_TRAIN = 6000
N_VAL   = 1000
N_TEST  = 1000

def load_captions(path=CAPTIONS_FILE):
    """
    Parse captions.txt where each line is:
        image.jpg,caption text
    or (Kaggle format):
        image.jpg#0  caption text

    Returns dict {filename: [caption, ...]} (up to 5 per image).
    """
    captions = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle both comma-separated and tab/space formats
            if "\t" in line:
                parts = line.split("\t", 1)
                fname = parts[0].split("#")[0].strip()
                cap   = parts[1].strip()
            else:
                parts = line.split(",", 1)
                if len(parts) < 2:
                    continue
                fname = parts[0].split("#")[0].strip()
                cap   = parts[1].strip()

            captions.setdefault(fname, []).append(cap)
    return captions


def clean_caption(text):
    """Lowercase, remove non-alpha characters except spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r" +", " ", text).strip()
    return text


def get_split_ids(captions, seed=42):
    """
    Deterministic 6000/1000/1000 split by image filename.
    Returns (train_ids, val_ids, test_ids) — lists of filenames.
    """
    all_ids = sorted(captions.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(all_ids)
    train_ids = all_ids[:N_TRAIN]
    val_ids   = all_ids[N_TRAIN:N_TRAIN + N_VAL]
    test_ids  = all_ids[N_TRAIN + N_VAL:]
    return train_ids, val_ids, test_ids


def save_split_ids(train_ids, val_ids, test_ids, split_dir=SPLIT_DIR):
    split_dir = Path(split_dir)
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        (split_dir / f"{name}_ids.txt").write_text("\n".join(ids))


def load_split_ids(split_dir=SPLIT_DIR):
    split_dir = Path(split_dir)
    def _read(name):
        p = split_dir / f"{name}_ids.txt"
        return p.read_text().splitlines() if p.exists() else None
    return _read("train"), _read("val"), _read("test")


def build_vocab(captions, train_ids, min_freq=1):
    """
    Build word2idx from training captions only.
    Reserves IDs 0-3 for special tokens.
    Returns (word2idx, idx2word).
    """
    counter = Counter()
    for fname in train_ids:
        for cap in captions.get(fname, []):
            counter.update(clean_caption(cap).split())

    word2idx = dict(SPECIAL_TOKENS)        # {<pad>:0, <start>:1, <end>:2, <unk>:3}
    for word, freq in sorted(counter.items()):
        if freq >= min_freq:
            word2idx[word] = len(word2idx)

    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def save_vocab(word2idx, idx2word, path=VOCAB_FILE):
    data = {
        "word2idx": word2idx,
        "idx2word": {int(k): v for k, v in idx2word.items()},
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"Vocab saved to {path}  ({len(word2idx)} tokens)")


def load_vocab(path=VOCAB_FILE):
    data = json.loads(Path(path).read_text())
    word2idx = data["word2idx"]
    idx2word = {int(k): v for k, v in data["idx2word"].items()}
    return word2idx, idx2word


def tokenize_caption(caption, word2idx, max_len):
    """
    Convert one caption string to a padded integer sequence.
    Output shape: (max_len,) int32
    """
    words  = clean_caption(caption).split()
    unk_id = word2idx["<unk>"]
    ids    = [word2idx["<start>"]]
    for w in words[:max_len - 2]:
        ids.append(word2idx.get(w, unk_id))
    ids.append(word2idx["<end>"])
    # pad to max_len
    ids += [word2idx["<pad>"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


def build_sequences(captions, ids, word2idx, max_len):
    """
    Build (image_id, tokenized_caption) pairs for a split.

    """
    img_ids, seqs = [], []
    for fname in ids:
        for cap in captions.get(fname, []):
            seq = tokenize_caption(cap, word2idx, max_len)
            img_ids.append(fname)
            seqs.append(seq)
    return img_ids, np.array(seqs, dtype=np.int32)

def run_preprocessing(max_len=35, min_freq=1):
    """
    Full pipeline: load → split → build vocab → tokenise → save artefacts.
    Returns (captions, word2idx, idx2word, train_ids, val_ids, test_ids).
    """
    print("Loading captions...")
    captions = load_captions()
    print(f"  {len(captions)} images, "
          f"{sum(len(v) for v in captions.values())} captions total")

    train_ids, val_ids, test_ids = get_split_ids(captions)
    save_split_ids(train_ids, val_ids, test_ids)
    print(f"  Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    word2idx, idx2word = build_vocab(captions, train_ids, min_freq=min_freq)
    save_vocab(word2idx, idx2word)
    print(f"  Vocabulary size: {len(word2idx)}")

    return captions, word2idx, idx2word, train_ids, val_ids, test_ids
