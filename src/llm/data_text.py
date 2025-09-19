import glob, os, io, math, random, json
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader

# 三种模式：char / bpe_simple / hf（需要 transformers）
from .tokenizer_simple import CharTokenizer, SimpleBPETokenizer

def load_text_files(glob_pattern: str, encoding: str = "utf-8") -> str:
    paths = sorted(glob.glob(glob_pattern, recursive=True))
    texts = []
    for p in paths:
        if os.path.isdir(p): 
            continue
        try:
            with io.open(p, "r", encoding=encoding, errors="ignore") as f:
                texts.append(f.read())
        except Exception:
            pass
    return "\n\n".join(texts)

class TextDataset(Dataset):
    def __init__(self, ids: torch.Tensor, block_size: int):
        self.ids = ids
        self.block_size = block_size

    def __len__(self):
        return max(0, self.ids.size(0) - self.block_size - 1)

    def __getitem__(self, idx):
        x = self.ids[idx: idx+self.block_size]
        y = self.ids[idx+1: idx+self.block_size+1]
        return x, y

def build_tokenizer(cfg):
    mode = cfg['tokenizer']['mode']
    if mode == 'char':
        return CharTokenizer()
    elif mode == 'bpe_simple':
        # 简易BPE：基于训练语料拟合
        return SimpleBPETokenizer(vocab_size=cfg['tokenizer'].get('vocab_size', 8000))
    elif mode == 'hf':
        from transformers import AutoTokenizer
        name = cfg['tokenizer'].get('pretrained_name', 'gpt2')
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # 适配接口
        class HFWrap:
            pad_id = property(lambda self: tok.pad_token_id)
            eos_id = property(lambda self: tok.eos_token_id)
            def encode(self, text): 
                return tok(text, add_special_tokens=False).input_ids
            def decode(self, ids): 
                return tok.decode(ids)
            def save(self, path): 
                tok.save_pretrained(path)
        return HFWrap()
    else:
        raise ValueError(f"Unknown tokenizer mode: {mode}")

def build_dataloaders(cfg):
    bs = cfg['train']['batch_size']
    ctx = cfg['model']['block_size']
    text_glob = cfg['data']['text_glob']

    raw_text = load_text_files(text_glob)
    assert len(raw_text) > 0, f"No text found by glob: {text_glob}"

    tokenizer = build_tokenizer(cfg)
    # 如果是 bpe_simple，需要先训练词表
    if cfg['tokenizer']['mode'] == 'bpe_simple':
        tokenizer.train(raw_text)

    ids = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)

    # 切分 train/val
    split = cfg['data'].get('train_val_split', 0.98)
    n = ids.size(0)
    n_train = int(n * split)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    train_ds = TextDataset(train_ids, ctx)
    val_ds   = TextDataset(val_ids, ctx)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2, drop_last=True)
    return tokenizer, train_loader, val_loader
