# 极简分词器：离线可用
from collections import Counter
import json

class CharTokenizer:
    def __init__(self):
        self.stoi = None
        self.itos = None
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    @property
    def pad_id(self): return self.pad_token_id
    @property
    def eos_id(self): return self.eos_token_id

    def fit(self, text: str):
        chars = sorted(set(text))
        self.stoi = {self.pad_token: self.pad_token_id, self.eos_token: self.eos_token_id}
        start = len(self.stoi)
        for i, ch in enumerate(chars):
            self.stoi[ch] = start + i
        self.itos = {i:s for s,i in self.stoi.items()}

    def encode(self, text: str):
        if self.stoi is None:
            self.fit(text)
        return [self.stoi.get(ch, self.eos_token_id) for ch in text]

    def decode(self, ids):
        return ''.join(self.itos.get(i, '') for i in ids)

    def save(self, path:str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos,
                       "pad_id": self.pad_token_id, "eos_id": self.eos_token_id}, f)

class SimpleBPETokenizer:
    """非常简化的BPE（演示用途，非高性能实现）。"""
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.stoi = None
        self.itos = None
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    @property
    def pad_id(self): return self.pad_token_id
    @property
    def eos_id(self): return self.eos_token_id

    def train(self, text: str):
        # 初始为字符级词表
        vocab = Counter(text)
        # 简化：不真正合并pair，只是保留高频字符，作为演示
        # 真正的BPE请使用 tokenizers 库
        chars = [c for c,_ in vocab.most_common(self.vocab_size-2)]
        self.stoi = {self.pad_token:self.pad_token_id, self.eos_token:self.eos_token_id}
        start = len(self.stoi)
        for i,ch in enumerate(chars):
            self.stoi[ch] = start+i
        self.itos = {i:s for s,i in self.stoi.items()}

    def encode(self, text: str):
        if self.stoi is None:
            self.train(text)
        return [self.stoi.get(ch, self.eos_token_id) for ch in text]

    def decode(self, ids):
        return ''.join(self.itos.get(i, '') for i in ids)

    def save(self, path:str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos,
                       "pad_id": self.pad_token_id, "eos_id": self.eos_token_id}, f)
